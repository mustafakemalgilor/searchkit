import abc
import concurrent.futures
import glob
import gzip
import multiprocessing
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
import hyperscan

from functools import cached_property
from collections import UserDict, UserList

from searchkit.log import log

RESULTS_QUEUE_TIMEOUT = 60
MAX_QUEUE_RETRIES = 10
RS_LOCK = multiprocessing.Lock()


def rs_locked(f):
    def _rs_locked_inner(*args, **kwargs):
        with RS_LOCK:
            return f(*args, **kwargs)

    return _rs_locked_inner


class FileSearchException(Exception):
    def __init__(self, msg):
        self.msg = msg


class SearchDefBase(object):

    def __init__(self, constraints=None):
        """
        @param constraints: an optional list of constraints to apply to
                            results.
        """
        self.id
        self.constraints = {c.id: c for c in constraints or {}}

    @cached_property
    def id(self):
        """ A unique identifier for this search definition. """
        return str(uuid.uuid4())


class SearchDef(SearchDefBase):

    def __init__(self, pattern, tag=None, hint=None,
                 store_result_contents=True, field_info=None, **kwargs):
        """
        Simple search definition.

        @param pattern: pattern or list of patterns to search for
        @param tag: optional user-friendly identifier for this search term.
                    This is useful for retrieving results.
        @param hint: optional pre-search term. If provided, this is expected to
                     match in order for the main search to be executed.
        @param store_result_contents: by default the content of a search result
                                      is saved but if it is not needed this
                                      can be set to False. This effectively
                                      makes the result True/False.
        @param field_info: optional ResultFieldInfo object
        """
        if type(pattern) != list:
            self.patterns = [re.compile(pattern)]
        else:
            self.patterns = []
            for _pattern in pattern:
                self.patterns.append(re.compile(_pattern))

        self.store_result_contents = store_result_contents
        self.tag = tag
        self.field_info = field_info
        self.hint = hint
        if hint:
            self.hint = re.compile(hint)

        self.sequence_def = None

        # do this last
        super().__init__(**kwargs)

    def link_to_sequence(self, sequence_def, tag):
        """
        If this search definition is part of a sequence, the parent
        SequenceSearchDef must link itself to this object.

        @param sequence_def: SequenceSearchDef object
        @param tag: SequenceSearchDef object tag for this section def
        """
        self.sequence_def = sequence_def
        self.tag = tag

    def run(self, line):
        """ Execute search patterns against line and return first match. """
        if self.hint:
            ret = self.hint.search(line)
            if not ret:
                return None

        ret = None
        for pattern in self.patterns:
            ret = pattern.match(line)
            if ret:
                break

        return ret


class HyperscanSearchDef(SearchDefBase):

    """
        Simple search definition, backed by hyperscan re.

        @param pattern: pattern or list of patterns to search for
        @param tag: optional user-friendly identifier for this search term.
                    This is useful for retrieving results.
        @param store_result_contents: by default the content of a search result
                                      is saved but if it is not needed this
                                      can be set to False. This effectively
                                      makes the result True/False.
        @param field_info: optional ResultFieldInfo object
    """

    # This is defined as class level variable
    # in order to be able to share hyperscan
    # databases between processes without having
    # to deal with `pickling`. Hyperscan instances
    # are multithread/multiprocess safe by default.
    hs_databases = {}

    def _compile_hs_db(self, patterns, *args, **kwargs):
        db = hyperscan.Database(*args, **kwargs)
        expressions, ids, flags = [], [], []
        group_names: dict[int, str] = {}
        for i, (group_name, expr) in enumerate(patterns.items()):
            ids.append(i)
            expressions.append(expr.encode())
            flags.append(0)
            group_names[i] = group_name

        db.compile(expressions=expressions, ids=ids, flags=flags)
        log.debug(
            "compiled hyperscan db for tag %s:"
            "%s, size: %d byte(s)",
            self.tag, db.info().decode(), db.size())
        return db

    def link_to_sequence(self, sequence_def, tag):
        """
        If this search definition is part of a sequence, the parent
        SequenceSearchDef must link itself to this object.

        @param sequence_def: SequenceSearchDef object
        @param tag: SequenceSearchDef object tag for this section def
        """
        self.sequence_def = sequence_def
        self.tag = tag

    def __init__(self, pattern, tag=None, hint=None,
                 store_result_contents=True, field_info=None,
                 **kwargs) -> None:
        self.tag = tag
        self.hint = hint  # not used atm.
        self.sequence_def = None
        self.field_info = field_info
        self.store_result_contents = store_result_contents

        if isinstance(pattern, list):
            self.patterns = dict(enumerate(pattern))
        else:
            self.patterns = {0: pattern}
        if self.id not in HyperscanSearchDef.hs_databases:
            dbs = {}
            dbs['prefilter'] = self._compile_hs_db(
                self.patterns, mode=hyperscan.HS_MODE_BLOCK)
            dbs['group'] = self._compile_hs_db(
                self.patterns, chimera=True, mode=hyperscan.CH_MODE_GROUPS)
            HyperscanSearchDef.hs_databases[self.id] = dbs
        # do this last
        super().__init__(**kwargs)

    def run(self, line):
        match_result = None

        class match_object:
            """An object type that mimics the
            re.match's return type.
            """
            def __init__(self, mg) -> None:
                self._line = mg[0]
                self._groups = {k: mg[k]
                                for k in set(list(mg.keys())) - set([0])}

            def groups(self):
                return self._groups

            def group(self, idx):
                if idx == 0:
                    return self._line
                return self.groups()[idx]

        def gm_on_match(_id, _start, _end, _flags, captured, _ctx):
            nonlocal match_result
            nonlocal line
            match_result = dict()

            for i, (_cap_flags, cap_from, cap_to) in enumerate(captured):
                match_result[i] = line[cap_from:cap_to]

        def on_match(_id, _start, _end, _flags, _ctx):
            HyperscanSearchDef.hs_databases[self.id]['group'].scan(
                line.encode(), gm_on_match)

        HyperscanSearchDef.hs_databases[self.id]['prefilter'].scan(
            line.encode(), on_match)

        return match_object(match_result) if match_result else None


class SequenceSearchDef(SearchDefBase):

    def __init__(self, start, tag, end=None, body=None, **kwargs):
        """
        Sequence search definition.

        A sequence must match a start and end with optional body in between.
        If no end is defined, the sequence ends with the start of the next or
        EOF.

        NOTE: sequences must not overlap. This is therefore not suitable for
        finding sequences generated by parallel/concurrent tasks.

        @param start: SearchDef object for matching start
        @param tag: tag used to identify this sequence definition
        @param end: optional SearchDef object for matching end
        @param body: optional SearchDef object for matching body
        """
        self.tag = tag
        self.s_start = start
        self.s_end = end
        self.s_body = body

        # make sure section defs have tags synced with this object
        for sd, _tag in {start: self.start_tag,
                         end: self.end_tag,
                         body: self.body_tag}.items():
            if sd:
                sd.link_to_sequence(self, _tag)

        self._mark = None
        # Each section matched gets its own id. Since each file is processed
        # using a separate process and memory is not shared, these values must
        # be unique to avoid collisions when results are aggregated.
        self._section_id = None
        self.completed_sections = []
        # do this last
        super().__init__(**kwargs)

    @property
    def start_tag(self):
        """ Tag used to identify start of section. """
        return "{}-start".format(self.tag)

    @property
    def end_tag(self):
        """ Tag used to identify end of section. """
        return "{}-end".format(self.tag)

    @property
    def body_tag(self):
        """ Tag used to identify body of section. """
        return "{}-body".format(self.tag)

    @property
    def current_section_id(self):
        """ ID of current section. A new id should be set after each
        completed section. """
        return self._section_id

    @property
    def started(self):
        """ Indicate a section sequence has been started. """
        return self._mark == 1

    def start(self):
        """ Indicate that a sequence start has been detected. """
        self._section_id = str(uuid.uuid4())
        log.debug("sequence %s started section %s (completed=%s)",
                  self.id, self.current_section_id,
                  len(self.completed_sections))
        self._mark = 1

    def reset(self):
        """ Used to restart a section. This is used e.g. if the start
        expression matches midway through a sequence (and before the end).
        """
        self._mark = 0

    def stop(self):
        """ Indicate that a sequence is complete. """
        self._mark = 0
        if self.current_section_id is None:
            raise FileSearchException("sequence section id is None")

        self.completed_sections.append(self.current_section_id)
        log.debug("sequence %s stopping section %s (completed=%s)",
                  self.id, self.current_section_id,
                  len(self.completed_sections))
        self._section_id = str(uuid.uuid4())

    def __repr__(self):
        return ("{}: current_section={}, started={}, completed_sections={}".
                format(self.__class__.__name__, self.current_section_id,
                       self.started, self.completed_sections))


class SequenceSearchResults(UserDict):

    def __init__(self):
        self.data = {}

    def add(self, result):
        id = result.sequence_id
        if id in self.data:
            self.data[id].append(result)
        else:
            self.data[id] = [result]

    def remove(self, id):
        if id in self.data:
            del self.data[id]


class ResultStoreBase(UserDict):

    def __init__(self):
        self.head = 0
        self.index = {}
        self.meta = {}
        self.data = {}

    def __getitem__(self, result_id):
        return self.data.get(result_id)

    def increment_head(self):
        """ Incrementing differs for proxied vs. raw types so we leave this to
        implementations to figure out. """
        self.head += 1

    @property
    def num_deduped(self):
        counters = self.meta.values()
        return sum(counters) - len(counters)

    def add(self, value):
        _id = self.index.get(value)
        if _id is not None:
            self.meta[_id] += 1
            return _id

        _id = self.head
        self.data[_id] = value
        self.meta[_id] = 1
        self.index[value] = _id
        self.increment_head()
        return _id


class ResultStoreSimple(ResultStoreBase):
    """ Store for use when sharing between processes is not needed. """


class ResultStoreParallel(ResultStoreBase):
    """ Store for use when sharing between processes is required. """

    def __init__(self, mgr):
        self._head = mgr.Value('i', 0)
        self.meta = mgr.dict()
        self.index = mgr.dict()
        self.data = mgr.dict()

    @rs_locked
    def __getitem__(self, result_id):
        return super().__getitem__(result_id)

    @property
    def head(self):
        return self._head.value

    def increment_head(self):
        self._head.value = self.head + 1

    @rs_locked
    def add(self, value):
        return super().add(value)

    @property
    @rs_locked
    def num_deduped(self):
        return super().num_deduped

    @rs_locked
    def unproxy_results(self):
        """
        Converts internal stores to unproxied types so they can be accessed
        once their manager is gone.
        """
        log.debug("unproxying results store (data=%s)", len(self.data))
        self._head = self._head.value
        self.data = self.data.copy()
        self.meta = self.meta.copy()
        self.index = self.index.copy()


class ResultFieldInfo(UserDict):

    def __init__(self, fields):
        """
        @param fields: list or dictionary of field names. If a dictionary is
                       provided, the values are expected to be functions that
                       the field value will be cast to. In other words these
                       should typically be standard or custom types.
        """
        if issubclass(fields.__class__, dict):
            self.data = fields
        else:
            self.data = {f: None for f in fields}

    def ensure_type(self, name, value):
        """
        If our fields have associated type functions, cast the value to
        its expected type.
        """
        if name not in self.data or self.data[name] is None:
            return value

        return self.data[name](value)

    def index_to_name(self, index):
        """ Retrieve a field name using the result group index. """
        for i, _field in enumerate(self.data):
            if index == i:
                return _field

        raise FileSearchException("field with index {} not found in mapping".
                                  format(index))


class SearchResultBase(UserList):

    def get(self, field):
        """
        Retrieve result part value by index or name.

        @param field: integer index of string field name.
        """
        for part in self.data:
            store_id = None
            if type(field) == str:
                if part['name'] == field:
                    store_id = part['store_id']
            elif part['idx'] == field:
                store_id = part['store_id']

            if store_id is not None:
                return self.results_store.get(store_id)

    def __getattr__(self, name):
        if name != 'field_info':
            if self.field_info and name in self.field_info:
                return self.get(name)

        raise AttributeError("'{}' object has no attribute '{}'".
                             format(self.__class__.__name__, name))

    def __iter__(self):
        """ Only return part values when iterating over this object. """
        for part in self.data:
            yield self.results_store.get(part['store_id'])

    def __repr__(self):
        r_list = ["{}='{}'".format(rp['idx'],
                                   self.results_store.get(rp['store_id']))
                  for rp in self.data]
        return ("ln:{} {} (section={})".
                format(self.linenumber, ", ".join(r_list),
                       self.section_id))


class SearchResultMinimal(SearchResultBase):

    def __init__(self, id, data, linenumber, source_id, tag,
                 sequence_id, sequence_section_id, field_info):
        """
        This is a minimised representation of a SearchResult object so as to
        reduce its size as much as possible before putting on the results
        queue.
        """
        self.id = id
        self.data = data
        self.linenumber = linenumber
        self.source_id = source_id
        self.tag = tag
        self.sequence_id = sequence_id
        self.section_id = sequence_section_id
        self.field_info = field_info
        self.results_store = None

    def register_results_store(self, store):
        """
        Register a ResultsStore with this result. This is used to re-register
        the store once the result has been received by the main process.

        @param store: ResultsStore object
        """
        self.results_store = store


class SearchResult(SearchResultBase):

    def __init__(self, linenumber, source_id, result, search_def,
                 results_store, sequence_section_id=None):
        """
        @param linenumber: line number that produced a match.
        @param source_id: data source id - resolves to a path in the
                          SearchCatalog.
        @param result: python.re.match object.
        @param search_def: SearchDef object.
        @param results_store: ResultsStore object
        @param sequence_section_id: if this result is part of a sequence the
                                    section ID must be provided.
        """
        self.results_store = results_store
        self.data = []
        self.linenumber = linenumber
        self.source_id = source_id
        self.tag = search_def.tag
        self.section_id = sequence_section_id
        self.sequence_id = None
        if search_def.sequence_def:
            if sequence_section_id is None:
                raise FileSearchException("sequence section result saved "
                                          "but no section id provided")

            self.sequence_id = search_def.sequence_def.id

        self.field_info = search_def.field_info

        if not search_def.store_result_contents:
            log.debug("store_contents is False - skipping save")
            return

        self.store_result(result)

    def store_result(self, result):
        num_groups = len(result.groups())
        # NOTE: this does not include group(0)
        if num_groups:
            # To reduce memory footprint, don't store group(0) i.e. the whole
            # line, if there are actual groups in the result.
            for i in range(1, num_groups + 1):
                self._save_part(i, result.group(i))
        else:
            log.debug("saving full search result which can lead to high "
                      "memory usage")
            self._save_part(0, result.group(0))

    @cached_property
    def id(self):
        """ Unique Result ID """
        id_string = "{}-{}-{}".format(uuid.uuid4(), self.source_id,
                                      self.linenumber)
        if self.sequence_id:
            id_string = "{}-{}-{}".format(id_string,
                                          self.sequence_id,
                                          self.section_id)
        return id_string

    def _save_part(self, part_index, value):
        name = None
        if value is not None and self.field_info:
            name = self.field_info.index_to_name(part_index - 1)
            value = self.field_info.ensure_type(name, value)

        store_id = self.results_store.add(value)
        self.data.append({'idx': part_index, 'store_id': store_id,
                          'name': name})

    @cached_property
    def export(self):
        """ Export the smallest possible representation of this object. """
        return SearchResultMinimal(self.id, self.data, self.linenumber,
                                   self.source_id, self.tag,
                                   self.sequence_id, self.section_id,
                                   self.field_info)


class SearchResultsCollection(UserDict):

    def __init__(self, search_catalog, results_store):
        self.search_catalog = search_catalog
        self.results_store = results_store
        self.reset()

    @property
    def data(self):
        results = {}
        for path, ids in self._results_by_path.items():
            results[path] = [self._results_by_id[id] for id in ids]

        return results

    @property
    def all(self):
        for r in self._results_by_id.values():
            yield r

    def reset(self):
        self._results_by_path = {}
        self._results_by_id = {}

    @property
    def files(self):
        return list(self._results_by_path.keys())

    def add(self, result):
        result.register_results_store(self.results_store)
        # resolve
        path = self.search_catalog.source_id_to_path(result.source_id)
        self._results_by_id[result.id] = result
        if path not in self._results_by_path:
            self._results_by_path[path] = [result.id]
        else:
            self._results_by_path[path].append(result.id)

    def find_by_path(self, path):
        """ Return results for a given path. """
        results = self._results_by_path.get(path, [])
        return [self._results_by_id[id] for id in results]

    def find_by_tag(self, tag, path=None):
        """ Return results matched by tag.

        @param tag: tag used to identify search results.
        @param path: optional path used to filter results to only include those
                     matched from a given path.
        """
        if path:
            paths = [path]
        else:
            paths = list(self._results_by_path.keys())

        results = []
        for path in paths:
            for result in self.find_by_path(path):
                if result.tag != tag:
                    continue

                results.append(result)

        return results

    def _get_all_sequence_results(self, path=None):
        """ Return a list of ids for all sequence match results.

        @param path: optionally filter results for a given path.
        """
        if path:
            paths = [path]
        else:
            paths = list(self._results_by_path.keys())

        sequences = []
        for path in paths:
            for result in self.find_by_path(path):
                if result.sequence_id is None:
                    continue

                sequences.append(result.id)

        return sequences

    def find_sequence_by_tag(self, tag, path=None):
        """ Find results for the sequence search(es) identified from tag.

        Returns a dictionary of "sections" i.e. complete sequences matched
        using associated SequenceSearchDef objects. Each section is a list of
        SearchResult objects representing start/body/end for that section.

        @param tag: tag used to identify sequence results.
        @param path: optionally filter results for a given path.
        """
        sections = {}
        for seq_obj in self.search_catalog.resolve_from_tag(tag):
            sections.update(self.find_sequence_sections(seq_obj, path))

        return sections

    def find_sequence_sections(self, sequence_obj, path=None):
        """ Find results for the given sequence search.

        Returns a dictionary of "sections" i.e. complete sequences matched
        using the associated SequenceSearchDef object. Each section is a list
        of SearchResult objects representing start/body/end for that section.

        @param sequence_obj: SequenceSearch object
        @param path: optionally filter results for a given path.
        """
        _results = {}
        for r in self._get_all_sequence_results(path=path):
            result = self._results_by_id[r]
            s_id = result.sequence_id
            if s_id != sequence_obj.id:
                continue

            section_id = result.section_id
            if section_id not in _results:
                _results[section_id] = []

            _results[section_id].append(result)

        return _results

    def __len__(self):
        """ Returns total number of search results. """
        _count = 0
        for f in self.files:
            _count += len(self.find_by_path(f))

        return _count


class LogrotateLogSort(object):

    def __call__(self, fname):
        """
        Sort contents of a directory by passing the function as the key to a
        list sort. Directory is expected to contain logfiles with extensions
        used by logrotate e.g. .log, .log.1, .log.2.gz etc.
        """
        filters = [r"\S+\.log$",
                   r"\S+\.log\.(\d+)$",
                   r"\S+\.log\.(\d+)\.gz?$"]
        for filter in filters:
            ret = re.compile(filter).match(fname)
            if ret:
                break

        # files that don't follow logrotate naming format go to the end.
        if not ret:
            # put at the end
            return 100000

        if len(ret.groups()) == 0:
            return 0

        return int(ret.group(1))


class SearchCatalog(object):

    def __init__(self, max_logrotate_depth=7):
        self.max_logrotate_depth = max_logrotate_depth
        self._source_ids = {}
        self._search_tags = {}
        self._simple_searches = {}
        self._sequence_searches = {}
        self._entries = {}

    def register(self, search, user_path):
        """
        Register a search against a path.

        The same search can be registered against more than one path.

        @param search: object implemented from SearchDefBase.
        @param user_path: directory or file path.
        """
        if search.tag is not None:
            if search.tag in self._search_tags:
                if search.id not in self._search_tags[search.tag]:
                    log.debug("one or more search tagged '%s' has already "
                              "been registered against path '%s'",
                              search.tag, user_path)
                    self._search_tags[search.tag].append(search.id)
            else:
                self._search_tags[search.tag] = [search.id]

        if isinstance(search, SequenceSearchDef):
            self._sequence_searches[search.id] = search
        else:
            self._simple_searches[search.id] = search

        for path in self._expand_path(user_path):
            if path in self._entries:
                entry = self._entries[path]
                entry['searches'].append(search)
            else:
                self._entries[path] = {'source_id': self._get_source_id(path),
                                       'path': path,
                                       'searches': [search]}

    def resolve_from_id(self, id):
        """ Resolve search definition from unique id. """
        if id in self._simple_searches:
            return self._simple_searches[id]

        return self._sequence_searches[id]

    def resolve_from_tag(self, tag):
        """ Resolve search definition from tag.

        Returns a list of resolved searches.
        """
        searches = []
        for id in self._search_tags[tag]:
            searches.append(self.resolve_from_id(id))

        return searches

    def _filtered_dir(self, contents, max_logrotate_depth=7):
        """ Filter contents of a directory. Directories are ignored and if any
        files look like logrotated log files they are sorted and only
        max_logrotate_depth are kept.
        """
        logrotated = {}
        new_contents = []
        for path in contents:
            if not os.path.isfile(path):
                continue

            ret = re.compile(r"(\S+)\.log\S*").match(path)
            if not ret:
                new_contents.append(path)
                continue

            fnamepfix = ret.group(1)
            if path.endswith('.log'):
                new_contents.append(fnamepfix + '.log')
            else:
                if fnamepfix not in logrotated:
                    logrotated[fnamepfix] = [path]
                else:
                    logrotated[fnamepfix].append(path)

        limit = max_logrotate_depth
        for logrotated in logrotated.values():
            capped = sorted(logrotated,
                            key=LogrotateLogSort())[:limit]
            new_contents += capped

        return new_contents

    def _expand_path(self, path):
        if os.path.isfile(path):
            return [path]
        elif os.path.isdir(path):
            return self._filtered_dir(os.listdir(path),
                                      self.max_logrotate_depth)

        return self._filtered_dir(glob.glob(path), self.max_logrotate_depth)

    def source_id_to_path(self, s_id):
        try:
            return self._source_ids[s_id]
        except KeyError:
            log.exception("ALL PATHS:")
            log.error('\n'.join(list(self._source_ids.keys())))

    def _get_source_id(self, path):
        for id, _path in self._source_ids.items():
            if _path == path:
                return id

        s_id = str(uuid.uuid4())
        while s_id in self._source_ids:
            log.error("source id %s already exists - trying again", s_id)
            s_id = str(uuid.uuid4())

        log.debug("path=%s source_id=%s", path, s_id)
        self._source_ids[s_id] = path
        return s_id

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        for entry in self._entries.values():
            yield entry


class SearchTask(object):

    def __init__(self, info, constraints_manager, results_queue,
                 results_store):
        self.info = info
        self.stats = SearchTaskStats()
        self.constraints_manager = constraints_manager
        self.results_queue = results_queue
        self.results_store = results_store

    @cached_property
    def id(self):
        return str(uuid.uuid4())

    @cached_property
    def search_defs_conditional(self):
        return [s_def for s_def in self.info['searches']
                if s_def.constraints]

    @cached_property
    def search_defs(self):
        all = {s_def: True for s_def in self.info['searches']}
        for s_def in all:
            if s_def in self.search_defs_conditional:
                all[s_def] = False

        return all

    def put_result(self, result):
        self.stats['results'] += 1
        max_tries = MAX_QUEUE_RETRIES
        while max_tries > 0:
            try:
                if max_tries == MAX_QUEUE_RETRIES:
                    self.results_queue.put_nowait(result.export)
                else:
                    self.results_queue.put(result.export,
                                           timeout=RESULTS_QUEUE_TIMEOUT)

                break
            except queue.Full:
                if max_tries == MAX_QUEUE_RETRIES:
                    msg = ("search task queue for '%s' is full - switching "
                           "to using blocking put with timeout")
                    log.info(msg, self.info['path'])
                else:
                    msg = ("search task queue for '%s' is full even after "
                           "waiting %ss - trying again")
                    log.warning(msg, self.info['path'],
                                RESULTS_QUEUE_TIMEOUT)

                max_tries -= 1

        if max_tries == 0:
            log.error("exceeded max number of retries (%s) to put results "
                      "data on the queue", MAX_QUEUE_RETRIES)

    def _simple_search(self, search_def, line, ln):
        """ Perform a simple search on line.

        @param search_def: SearchDef object
        @param line: current line (string)
        @param ln: current line number
        """
        ret = search_def.run(line)
        if not ret:
            return

        self.put_result(SearchResult(ln, self.info['source_id'], ret,
                                     search_def, self.results_store))

    def _sequence_search(self, seq_def, line, ln, sequence_results):
        """ Perform a sequence search on line.

        @param seq_def: SequenceSearchDef object
        @param line: current line (string)
        @param ln: current line number
        @param sequence_results: SequenceSearchResults object
        """
        ret = seq_def.s_start.run(line)
        # if the ending is defined and we match a start while
        # already in a section, we start again.
        if seq_def.s_end and seq_def.started:
            if ret:
                # reset and start again
                sequence_results.remove(seq_def.id)
                seq_def.reset()
            else:
                ret = seq_def.s_end.run(line)

        if ret:
            if not seq_def.started:
                s_term = seq_def.s_start
                seq_def.start()
                section_id = seq_def.current_section_id
            else:
                s_term = seq_def.s_end
                section_id = seq_def.current_section_id
                seq_def.stop()
                # if no end is defined then we don't bother storing
                # the result, just complete the section and start
                # the next.
                if seq_def.s_end is None:
                    s_term = seq_def.s_start
                    seq_def.start()
                    section_id = seq_def.current_section_id

            sequence_results.add(SearchResult(ln, self.info['source_id'], ret,
                                              s_term, self.results_store,
                                              sequence_section_id=section_id))
        elif seq_def.started and seq_def.s_body:
            section_id = seq_def.current_section_id
            ret = seq_def.s_body.run(line)
            if ret:
                sequence_results.add(SearchResult(
                                            ln, self.info['source_id'],
                                            ret, seq_def.s_body,
                                            self.results_store,
                                            sequence_section_id=section_id))

    def _process_sequence_results(self, sequence_results, current_ln):
        """
        Perform post processing to sequence search results.

        @param sequence_results: SequenceSearchResults object
        @param current_ln: number of the last line to be read from file
        """
        # If a sequence ending definition is provided and we reached EOF
        # while a sequence is started, complete the sequence if s_end
        # matches an empty string. If s_end is not defined we just go ahead
        # and complete the section.
        filter_section_id = {}
        for s_def in self.search_defs:
            if type(s_def) != SequenceSearchDef:
                continue

            seq_def = s_def
            if not seq_def.started:
                continue

            if seq_def.s_end is None:
                continue

            ret = seq_def.s_end.run('')
            if ret:
                section_id = seq_def.current_section_id
                r = SearchResult(current_ln + 1, self.info['source_id'], ret,
                                 seq_def.s_end,
                                 self.results_store,
                                 sequence_section_id=section_id)
                sequence_results.add(r)
            else:
                if seq_def.id not in filter_section_id:
                    filter_section_id[seq_def.id] = []

                filter_section_id[seq_def.id].append(
                    seq_def.current_section_id)

        if len(sequence_results) < 1:
            log.debug("no sequence results to process")
            return

        log.debug("filtering sections: %s", filter_section_id)
        # Now add sequence results to main results list, excluding any
        # incomplete sections.
        for s_results in sequence_results.values():
            for r in s_results:
                if filter_section_id:
                    if r.sequence_id is None:
                        continue

                    seq_id = r.sequence_id
                    if seq_id in filter_section_id:
                        if r.section_id in filter_section_id[seq_id]:
                            continue

                self.put_result(r)

    def _run_search(self, fd):
        """
        @param fd: open file descriptor
        """
        self.stats.reset()
        sequence_results = SequenceSearchResults()
        search_ids = set([s.id for s in self.search_defs])
        offset = self.constraints_manager.apply_global(search_ids, fd)
        log.debug("starting search of %s (offset=%s, pos=%s)", fd.name, offset,
                  fd.tell())
        runnable = {s.id: _runnable
                    for s, _runnable in self.search_defs.items()}
        ln = 0
        # NOTE: line numbers start at 1 hence offset + 1
        for ln, line in enumerate(fd, start=offset + 1):
            # This could be helpful to show progress for large files
            if ln % 100000 == 0:
                log.debug("%s lines searched in %s", ln, fd.name)

            self.stats['lines_searched'] += 1
            if type(line) == bytes:
                line = line.decode("utf-8")

            for s_def in self.search_defs:
                if not runnable[s_def.id]:
                    if not self.constraints_manager.apply_single(s_def, line):
                        continue

                    # enable from here on in
                    runnable[s_def.id] = True

                if type(s_def) == SequenceSearchDef:
                    self._sequence_search(s_def, line, ln, sequence_results)
                else:
                    self._simple_search(s_def, line, ln)

        self._process_sequence_results(sequence_results, ln)
        log.debug("completed search of %s lines", self.stats['lines_searched'])
        if self.search_defs_conditional:
            msg = "constraints stats {}:".format(fd.name)
            for sd in self.search_defs_conditional:
                if sd.constraints:
                    for c in sd.constraints.values():
                        msg += "\n  id={}: {}".format(c.id, c.stats())

            log.debug(msg)

        log.debug("run search complete for path %s", fd.name)
        return self.stats

    def failed(self, exc):
        """ This should be called if the task failed to execute. """
        log.error("search task failed for path=%s with exception %s",
                  self.info['path'], exc)

    def execute(self):
        stats = SearchTaskStats()
        path = self.info['path']
        if os.path.getsize(path) == 0:
            log.debug("filesearcher: zero-length file %s - skipping search",
                      path)
            return stats

        log.debug("starting execution on path %s (searches=%s)", path,
                  len(self.search_defs))
        try:
            # first assume compressed then plain
            with gzip.open(path, 'rb') as fd:
                try:
                    # test if file is gzip
                    fd.read(1)
                    fd.seek(0)
                    stats = self._run_search(fd)
                except OSError:
                    with open(path, 'rb', buffering=1024*1024*1024) as fd:
                        stats = self._run_search(fd)
        except UnicodeDecodeError:
            log.exception("")
            # ignore the file if it can't be decoded
            log.debug("caught UnicodeDecodeError for path %s - skipping",
                      path)
        except EOFError as e:
            log.exception("")
            msg = ("an exception occurred while searching {} - {}".
                   format(path, e))
            raise FileSearchException(msg) from e
        except Exception as e:
            log.exception("")
            msg = ("an unexpected exception occurred while searching {} - {}".
                   format(path, e))
            raise FileSearchException(msg) from e

        log.debug("finished execution on path %s", path)
        return stats


class SearchTaskStats(UserDict):

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = {'searches': 0,
                     'searches_by_job': [],
                     'lines_searched': 0,
                     'jobs_completed': 0,
                     'total_jobs': 0,
                     'results': 0,
                     'num_deduped': 0}

    def update(self, stats):
        if not stats:
            return

        for key, val in stats.items():
            self.data[key] += val

    def __repr__(self):
        return ', '.join([f"{k}={v}" for k, v in self.data.items()])


class SearcherBase(abc.ABC):

    @abc.abstractproperty
    def files(self):
        """ Returns a list of files we will be searching. """

    @abc.abstractproperty
    def num_parallel_tasks(self):
        """
        Returns an integer representing the maximum number of tasks we can
        run in parallel. This will typically be bound by the number of
        cpu threads available.
        """

    @abc.abstractmethod
    def add(self, searchdef):
        """
        Add a search criterea.

        @param searchdef: SearchDef object
        """

    @abc.abstractmethod
    def run(self):
        """
        Execute all searches.
        """


class SearchConstraintsManager(object):

    def __init__(self, search_catalog):
        self.search_catalog = search_catalog
        self.global_constraints = []
        self.global_restrictions = set()

    def apply_global(self, search_ids, fd):
        """ Apply any global constraints to the entire file. """
        offset = 0
        if not self.global_constraints:
            log.debug("no global constraint to apply to %s", fd.name)
            return offset

        if self.global_restrictions.intersection(search_ids):
            log.debug("skipping global constraint for %s", fd.name)
            return offset

        for c in self.global_constraints:
            log.debug("applying task global constraint %s to %s", c.id,
                      fd.name)
            _offset = c.apply_to_file(fd)
            if _offset is not None:
                return _offset

        return offset

    def apply_single(self, searchdef, line):
        """
        Apply any constraints for this searchdef to the give line.
        """
        if not searchdef.constraints:
            return True

        for c in searchdef.constraints.values():
            if c.apply_to_line(line):
                continue

            return False

        return True


class FileSearcher(SearcherBase):

    def __init__(self, max_parallel_tasks=8, max_logrotate_depth=7,
                 constraint=None):
        """
        @param max_parallel_tasks: max number of search tasks that can run in
                                   parallel.
        @param max_logrotate_depth: used by SearchCatalog to filter logfiles
                                    based on their name if it matches a
                                    logrotate format and want to constrain how
                                    much history we search.
        @param constraint: constraint to be used with this
                                   searcher that applies to all files searched.
        """
        self.max_parallel_tasks = max_parallel_tasks
        self._stats = SearchTaskStats()
        self.catalog = SearchCatalog(max_logrotate_depth)
        self.constraints_manager = SearchConstraintsManager(self.catalog)
        if constraint:
            self.constraints_manager.global_constraints.append(constraint)

    @property
    def files(self):
        return [e['path'] for e in self.catalog]

    def resolve_source_id(self, source_id):
        return self.catalog.source_id_to_path(source_id)

    def add(self, searchdef, path, allow_global_constraints=True):
        """
        Add a search definition.

        @param searchdef: a SearchDef or SequenceSearchDef object.
        @param path: path we want to search. this can be a file, dir or glob.
        @param allow_global_constraints: boolean determining whether we want
                                         any global constraints available to be
                                         applied to this path.
        """
        if not allow_global_constraints:
            self.constraints_manager.global_restrictions.add(searchdef.id)

        self.catalog.register(searchdef, path)

    @property
    def num_parallel_tasks(self):
        if self.max_parallel_tasks == 0:
            cpus = 1  # i.e. no parallelism
        else:
            cpus = min(self.max_parallel_tasks, os.cpu_count())

        return min(len(self.files) or 1, cpus)

    @property
    def stats(self):
        """
        Provide stats for the last search run.

        @return: SearchTaskStats object
        """
        return self._stats

    def _get_results(self, results, queue, event, stats):
        """
        Collect results from all search task processes.

        @param results: SearchResultsCollection object.
        @param queue: results queue used for this search session.
        @param event: event object used to notify this thread to stop.
        @param stats: SearchTaskStats object
        """
        log.debug("fetching results from worker queues")

        while True:
            if not queue.empty():
                results.add(queue.get())
            elif event.is_set():
                log.debug("exiting results thread")
                break
            else:
                log.debug("total %s results received, %s/%s jobs completed - "
                          "waiting for more", len(results),
                          stats['jobs_completed'], stats['total_jobs'])
                # yield
                time.sleep(0.1)

        log.debug("stopped fetching results (total received=%s)", len(results))

    def _purge_results(self, results, queue, expected):
        """
        Purge results from all search task processes.

        @param results: SearchResultsCollection object.
        @param queue: results queue used for this search session.
        @param expected: number of results we expect to receive. this is used
                         to do a final sweep once all search tasks are complete
                         to ensure all results have been collected.
        """
        log.debug("purging results (expected=%s)", expected)

        while True:
            if not queue.empty():
                results.add(queue.get())
            elif expected > len(results):
                try:
                    r = queue.get(timeout=RESULTS_QUEUE_TIMEOUT)
                    results.add(r)
                except queue.Empty:
                    log.info("timeout waiting > %s secs to receive results - "
                             "expected=%s, actual=%s", RESULTS_QUEUE_TIMEOUT,
                             expected, len(results))
            else:
                break

        log.debug("stopped purging results (total received=%s)",
                  len(results))

    def _create_results_thread(self, results, queue, stats):
        log.debug("creating results queue consumer thread")
        event = threading.Event()
        event.clear()
        t = threading.Thread(target=self._get_results,
                             args=[results, queue, event, stats])
        return t, event

    def _stop_results_thread(self, thread, event):
        log.debug("joining/stopping queue consumer thread")
        event.set()
        thread.join()
        log.debug("consumer thread stopped successfully")

    def _ensure_worker_processes_killed(self):
        """
        For some reason it is sometimes possible to for pool termination to
        hang indefinitely because one or more worker process fails to
        terminate. This method ensures that all extant worker child processes
        are killed so that pool termination is guaranteed to complete.
        """
        log.debug("ensuring all pool workers killed")
        worker_pids = []
        for child in multiprocessing.active_children():
            if type(child) == multiprocessing.context.ForkProcess:
                if 'ForkProcess' in child.name:
                    worker_pids.append(child.pid)

        ps_out = subprocess.check_output(['ps', '-opid', '--no-headers',
                                          '--ppid',
                                          str(os.getpid())], encoding='utf8')
        child_pids = [int(line.strip()) for line in ps_out.splitlines()]
        log.debug("process has child pids: %s", child_pids)
        for wpid in worker_pids:
            if int(wpid) not in child_pids:
                log.error("worker pid %s no longer a child of this process "
                          "(%s)", wpid, os.getpid())
                continue

            try:
                log.debug('sending SIGKILL to worker process %s', wpid)
                os.kill(wpid, signal.SIGILL)
            except Exception:
                log.debug('worker process %s already killed', wpid)

    def _run_single(self, results, results_store):
        """ Run a single search using this process.

        @param results: SearchResultsCollection object
        """
        queue = multiprocessing.Queue()
        for info in self.catalog:
            task = SearchTask(info,
                              constraints_manager=self.constraints_manager,
                              results_queue=queue,
                              results_store=results_store)
            self.stats.update(task.execute())

        self.stats['jobs_completed'] = 1
        self.stats['total_jobs'] = 1
        self._purge_results(results, queue, self.stats['results'])

    def _run_mp(self, mgr, results, results_store):
        """ Run searches in parallel.

        @param mgr: multiprocessing.Manager object
        @param results: SearchResultsCollection object
        """
        queue = mgr.Queue()
        results_thread, event = self._create_results_thread(results, queue,
                                                            self.stats)
        results_thread_started = False
        try:
            num_workers = self.num_parallel_tasks
            with concurrent.futures.ProcessPoolExecutor(
                                    max_workers=num_workers) as executor:
                jobs = {}
                for info in self.catalog:
                    c_mgr = self.constraints_manager
                    task = SearchTask(info,
                                      constraints_manager=c_mgr,
                                      results_queue=queue,
                                      results_store=results_store)
                    job = executor.submit(task.execute)
                    jobs[job] = info['path']
                    self.stats['total_jobs'] += 1

                log.debug("filesearcher: syncing %s job(s)", len(jobs))
                results_thread.start()
                results_thread_started = True
                try:
                    for future in concurrent.futures.as_completed(jobs):
                        self.stats.update(future.result())
                        self.stats['jobs_completed'] += 1
                except concurrent.futures.process.BrokenProcessPool as exc:
                    msg = ("one or more worker processes has died - "
                           "aborting search")
                    raise FileSearchException(msg) from exc

                log.debug("all workers synced")
                # double check nothing is running anymore
                for job, path in jobs.items():
                    log.debug("worker for path '%s' has state: %s", path,
                              repr(job))
                    if job.running():
                        log.info("job for path '%s' still running when "
                                 "not expected to be", path)

                self._stop_results_thread(results_thread, event)
                results_thread = None
                log.debug("purging remaining results (expected=%s, "
                          "remaining=%s)", self.stats['results'],
                          self.stats['results'] - len(results))
                self._purge_results(results, queue, self.stats['results'])

                self._ensure_worker_processes_killed()
                log.debug("terminating pool")
        finally:
            if results_thread is not None and results_thread_started:
                self._stop_results_thread(results_thread, event)

    def run(self):
        """ Run all searches.

        @return: SearchResultsCollection object
        """
        log.debug("filesearcher: starting")
        self.stats.reset()
        if len(self.catalog) == 0:
            log.debug("catalog is empty - nothing to run")
            return SearchResultsCollection(self.catalog, ResultStoreSimple())

        self.stats['searches'] = sum([len(p['searches'])
                                      for p in self.catalog])
        self.stats['searches_by_job'] = [len(p['searches'])
                                         for p in self.catalog]
        if len(self.files) > 1:
            log.debug("running searches (parallel=True)")
            with multiprocessing.Manager() as mgr:
                rs = ResultStoreParallel(mgr)
                results = SearchResultsCollection(self.catalog, rs)
                self._run_mp(mgr, results, rs)
                self.stats['num_deduped'] = rs.num_deduped
                rs.unproxy_results()
        else:
            log.debug("running searches (parallel=False)")
            rs = ResultStoreSimple()
            results = SearchResultsCollection(self.catalog, rs)
            self._run_single(results, rs)
            self.stats['num_deduped'] = rs.num_deduped

        log.debug("filesearcher: completed (%s)", self.stats)
        return results
