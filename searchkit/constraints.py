import abc
import re
import uuid
import bisect

from datetime import datetime, timedelta
from functools import cached_property

from searchkit.log import log


class TimestampMatcherBase(object):
    """
    Match start of line timestamps in a standard way.

    Files containing lines starting with timestamps allow us to find a line
    that is before/after a specific time. This class is implemented to provides
    a common way to identify timestamps of varying format.
    """

    # used when converting a string to datetime.datetime
    DEFAULT_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, line):
        self.result = None
        for expr in self.patterns:
            ret = re.match(expr, line)
            if ret:
                self.result = ret
                break
        else:
            log.debug("failed to identify constraint datetime")

    @property
    @abc.abstractmethod
    def patterns(self):
        """
        List of regex patterns used to match a timestamp at the start of lines.

        Patterns *must* use named groups according to types i.e. year,
        month etc. See https://docs.python.org/3/library/re.html for format
        options.

        If the format of the timestamp is non-standard and the result needs
        post-processing before being used, a property with the group name
        can be added to implementations of this class and that will be used
        rather than extracting the value directly from the result.
        """

    @property
    def matched(self):
        """ Return True if a timestamp has been matched. """
        return self.result is not None

    @property
    def strptime(self):
        """
        Converts the extracted timestamp into a datetime.datetime object.

        Group names are extracted directly from the result unless an override
        property has been defined.

        @return: datetime.datetime object
        """
        vals = {}
        for key in ['day', 'month', 'year', 'hours', 'minutes', 'seconds']:
            if hasattr(self, key):
                vals[key.rstrip('s')] = int(getattr(self, key))
            else:
                vals[key.rstrip('s')] = int(self.result.group(key))

        return datetime(**vals)


class ConstraintBase(abc.ABC):

    @cached_property
    def id(self):
        """
        A unique identifier for this constraint.
        """
        return uuid.uuid4()

    @abc.abstractmethod
    def apply_to_line(self, line):
        """
        Apply constraint to a single line.

        @param fd: file descriptor
        """

    @abc.abstractmethod
    def apply_to_file(self, fd):
        """
        Apply constraint to an entire file.

        @param fd: file descriptor
        """

    @abc.abstractmethod
    def stats(self):
        """ provide runtime stats for this object. """

    @abc.abstractmethod
    def __repr__(self):
        """ provide string repr of this object. """


class BinarySeekSearchBase(ConstraintBase):
    """
    Provides a way to seek to a point in a file using a binary search and a
    given condition.
    """

    def __init__(self, allow_constraints_for_unverifiable_logs=True):
        self.allow_unverifiable_logs = allow_constraints_for_unverifiable_logs

    @abc.abstractmethod
    def extracted_datetime(self, line):
        """
        Extract timestamp from start of line.

        @param line: text line to extract a datetime from.
        @return: datetime.datetime object or None
        """

    @abc.abstractproperty
    def _since_date(self):
        """ A datetime.datetime object representing the "since" date/time """

    def _line_date_is_valid(self, extracted_datetime):
        """
        Validate if the given line falls within the provided constraint. In
        this case that's whether it has a datetime that is >= to the "since"
        date.
        """
        ts = extracted_datetime
        if ts is None:
            # log.info("s:%s: failed to extract datetime from "
            #          "using expressions %s - assuming line is not valid",
            #          unique_search_id, ', '.join(self.exprs))
            return False

        if ts < self._since_date:
            # log.debug("%s < %s at (%s) i.e. False", ts, self._since_date,
            #           line[-3:].strip())
            return False

        # log.debug("%s >= %s at (%s) i.e. True", ts, self._since_date,
        #           line[-3:].strip())

        return True


class NoLogsFoundSince(Exception):
    """Raised when a log file contains proper timestamps but
    no log lines after the since date."""


class NoDateFoundInLogs(Exception):
    """Raised when a log file does not contain any line with
    date suitable to specified date format"""


class DateSearchFailedAtOffset(Exception):
    """Raised when searcher has encountered a line with no date
    and performed forward-backward searches, but still yet, could
    not found a line with date."""


class LogFileDateSinceOffsetSeeker:
    """This class allows user to perform `since` date lookups with
    file offsets. This is useful for performing line-based binary
    date searches on a log file.

    The class implements __len__ and __getitem__ methods in order to
    behave like a list. When __getitem__ is called with ane offset,
    the algorithm locates the rightmost and leftmost line feed (`\n`)
    to form a line. Assume the following file contents:

    13:15 AAAAAA\n13:16 BBBBBBB\n13:17 CCCCCC

    ... and let's assume that the __getitem__ function is called with
    offset `19`:

    13:15 AAAAAA\n13:16 BBBBBBB\n13:17 CCCCCC
                        ^19

    The algorithm first will read SEEK_HORIZON bytes forward, starting
    from offset `19`, and then try to find the first line feed:

    13:15 AAAAAA\n13:16 BBBBBBB\n13:17 CCCCCC
                        ^19     ^r-lf

    Consequently, the algorithm will seek SEEK_HORIZON bytes backward,
    starting from offset `19`, read SEEK_HORIZON bytes and then try to
    find the first line feed, scanning in reverse:

    13:15 AAAAAA\n13:16 BBBBBBB\n13:17 CCCCCC
                ^l-lf   ^19     ^r-lf

    Then, the algoritm will extract the characters between l-lf and r-lf
    to form a line. The line will be checked against the date matcher
    to extract the date. If the date matcher yields a valid date, the
    __getitem__ function will return that date. Otherwise, the search will
    be extended to other nearby lines, prioritizing the lines prior to the
    current, until either:

        - a line with timestamp found, or
        - MAX_*_FALLBACK_LINES has reached.
    """

    # Amount of characters to read while searching
    SEEK_HORIZON = 256

    # How many times we can expand the search
    # horizon while trying to find a line feed.
    # This means the search will read SEEK_HORIZON
    # times MAX_SEEK_HORIZON_EXPAND bytes in total
    # when a line feed character is not found.
    MAX_SEEK_HORIZON_EXPAND = 100

    # How many lines should we search forwards utmost
    # when the algorithm encounters lines with no date.
    MAX_FWD_FALLBACK_LINES = 500

    # How many lines should we search backwards utmost
    # when the algorithm encounters lines with no date.
    MAX_RWD_FALLBACK_LINES = 500

    def __init__(self, fd, c) -> None:
        self.file = fd
        self.constraint = c
        self.line_info = None
        self.found_any_date = False

    def find_delimiter_rev(
        self,
        file,
        start_offset,
        horizon,
        token=b"\n",
        max_depth=MAX_SEEK_HORIZON_EXPAND,
    ):
        """Find `token` in `file` starting from `start_offset` and backing off
        `horizon`bytes on each iteration for maximum of `max_depth` times.

        Args:
            file (file): File descriptor, open in read mode
            start_offset (int): start offset of search
            horizon (int): Amount of bytes to be processed on each step.
            token (str, optional): Search token. Defaults to `\n`.
            max_depth (int, optional): Maximum amount of search iterations.
                                       Defaults to 100.

        Returns:
            (bool, int, int): true, <token offset>, <search reach point>
            if found
            (False, None, None): Token could not be found
        """

        pos = -horizon
        while max_depth > 0:
            offset = start_offset + pos
            offset = offset if offset > 0 else 0
            rsize = horizon
            if start_offset + pos <= 0:
                rsize = rsize + (start_offset + pos)

            file.seek(offset)
            chunk = file.read(rsize)
            if not chunk or len(chunk) == 0:
                return (False, 0, 0)

            coff = chunk.rfind(token)

            if coff == -1:
                pos = pos - len(chunk)
                max_depth -= 1
                if start_offset + pos < 0:
                    return (False, 0, start_offset + pos)
                continue

            reach = start_offset + pos - len(chunk)
            return (True, offset + coff, reach if reach > 0 else 0)

        return (False, None, None)

    def find_delimiter(
        self,
        file,
        start_offset,
        horizon,
        token=b"\n",
        max_depth=MAX_SEEK_HORIZON_EXPAND,
    ):
        """Find `token` in `file` starting from `start_offset` and moving
        forward `horizon` bytes on each iteration for maximum of `max_depth`
        times.

        Args:
            file (file): File descriptor, open in read mode
            start_offset (int): start offset of search
            horizon (int): Amount of bytes to be processed on each step.
            token (str, optional): Search token. Defaults to "\n".
            max_depth (int, optional): Maximum amount of search iterations.
            Defaults to 100.

        Returns:
            int: offset of `token`, if token is found
            None: Token could not be found
        """

        pos = 0
        file.seek(start_offset)
        while max_depth > 0:
            chunk = file.read(horizon)

            if not chunk or len(chunk) == 0:
                return (False, len(self), len(self))

            coff = chunk.find(token)
            if coff == -1:
                pos = pos + len(chunk)
                max_depth -= 1
                continue
            return (True,
                    start_offset + pos + coff,
                    (start_offset + pos + len(chunk)))
        return (False, None, None)

    def try_find_line_w_date(self, epicenter, slf_off=None, elf_off=None):
        """Try to find a line at `epicenter`.

        This function will perform a horizon-expand search to locate the
        nearest line feed (`\n`) (fwd_line_feed) where pos(`\n`) is >
        start_offset, and then will locate the nearest line feed (`\n`)
        (rwd_line_feed) where pos (`\n`) is < start_offset (i.e. backward),
        and will form a line (rwd_line_feed, fwd_line_feed) and check whether
        the line contains a date compatible with the timestamp matcher.

        Args:
            epicenter (int): Search start offset
            slf_off (int, optional): Starting line feed offset, if known.
            Defaults to None.
            elf_off (int, optional): Ending line feed offset, if known.
            Defaults to None.

        Raises:
            ValueError: when ending line feed offset could not be found
            ValueError: when starting line feed offset could not be found

        Returns:
            (date, (bool, int, int), (bool, int, int)): Date of the line at
            offset `epicenter` if found, (start_lf_found, start_lf_offset,
            start_lf_search_reach_offset), (end_lf_found, end_lf_offset,
            end_lf_search_reach_offset)
        """
        log.debug("    > EPICENTER: %d", epicenter)
        if not elf_off:
            fdf = self.find_delimiter(
                self.file, epicenter, LogFileDateSinceOffsetSeeker.SEEK_HORIZON
            )
        else:
            fdf = (True, elf_off, elf_off)

        found_elf, elf_off, _ = fdf

        if elf_off is None:
            raise ValueError("Could not find ending line feed offset")

        if not slf_off:
            fdr = self.find_delimiter_rev(
                self.file, epicenter, LogFileDateSinceOffsetSeeker.SEEK_HORIZON
            )
        else:
            fdr = (True, slf_off, slf_off)

        found_slf, slf_off, _ = fdr

        if slf_off is None:
            raise ValueError()

        assert slf_off <= len(self)
        assert slf_off >= 0
        assert elf_off <= len(self)
        assert elf_off >= 0

        line_start = slf_off + 1 if found_slf else slf_off
        line_end = elf_off
        read_amount = line_end - line_start
        log.debug(
            "    > SLF: (%s, %s), ELF: (%s, %s), READ_AMOUNT: %d",
            found_slf,
            slf_off,
            found_elf,
            elf_off,
            read_amount,
        )
        self.file.seek(line_start)
        line = self.file.read(read_amount)
        log.debug(
            "    > LINE:"
            + line.decode("utf-8")
            + f", DATE FORMAT: {self.constraint.date_format}"
        )
        date = self.constraint.extracted_datetime(line)
        log.debug("    > RET: %s, %s, %s", date, fdf, fdr)
        return (date, fdf, fdr)

    def try_find_line_w_date_for(
        self, how_many_lines, start_offset, prev_offset=None, forwards=False
    ):
        """Try to fetch a line with date, starting from `start_offset`.

        The algorithm will try to fetch a new line searching for a valid date
        for a maximum of `how_many_lines` times. The lines will be fetched from
        either prior or after `start_offset`, depending on the value of the
        `forwards` parameter.

        If `prev_offset` parameter is used, the value will be used as either
        fwd_line_feed or rwd_line_feed position depending on the value of the
        `forwards` parameter.

        Args:
            how_many_lines (int): How many lines will be checked utmost.
            start_offset (int): Where to begin searching
            prev_offset (int, optional): Offset of the fwd_line_feed,
            or rwd_line_feed if known. Defaults to None.
            forwards (bool, optional): Search forwards, or backwards.
            Defaults to False (forwards).

        Returns:
            (date, (bool, int, int), (bool, int, int)): Date of the line at
            offset `epicenter` if found, (start_lf_found, start_lf_offset,
            start_lf_search_reach_offset), (end_lf_found, end_lf_offset,
            end_lf_search_reach_offset)
        """
        date, fwd_r, rwd_r = None, (None, None, None), (None, None, None)
        offset = start_offset
        while date is None and how_many_lines > 0:
            (date, fwd_r, rwd_r) = self.try_find_line_w_date(
                offset,
                prev_offset if forwards else None,
                prev_offset if not forwards else None,
            )

            _, rwd_offset, _ = rwd_r
            _, fwd_offset, _ = fwd_r

            log.debug(
                "    TRY_FETCH %d %d %d >:" "on line -> %s",
                how_many_lines,
                rwd_offset,
                fwd_offset,
                self.__debug_seek_and_read_line(rwd_offset, fwd_offset),
            )

            prev_offset = fwd_offset if forwards else rwd_offset
            offset = fwd_offset + 1 if forwards else rwd_offset - 1
            if offset < 0 or offset > len(self):
                log.debug("    TRY_FETCH EXIT EOF/SOF")
                break
            how_many_lines -= 1
        return (date, fwd_r, rwd_r)

    def __len__(self):
        orig = self.file.tell()
        eof = self.file.seek(0, 2)
        self.file.seek(orig)
        return eof

    def __debug_seek_and_read_line(self, line_soff, line_eoff):
        orig = self.file.tell()
        self.file.seek(line_soff)
        line = self.file.read(line_eoff - line_soff)
        self.file.seek(orig)
        return line

    def __getitem__(self, item):
        log.debug("-------------------------------------------")
        log.debug("-------------------------------------------")
        log.debug("-------------------------------------------")
        log.debug("-------------------------------------------")
        log.debug("LOOKUP INDEX: %d", item)

        date = None

        # First, try to find the line as usual.
        # That'll be the most common scenario we'll
        # encounter.
        (
            date,
            (fwdlf_found, fwdlf_offset, _),
            (rwdlf_found, rwdlf_offset, _),
        ) = self.try_find_line_w_date(item)

        orig_rwdlf_offset = rwdlf_offset
        orig_fwdlf_offset = fwdlf_offset

        # Try to search backwards first.
        if date is None:
            log.debug("######### BACKWARDS SEARCH START #########")
            (
                date,
                (fwdlf_found, fwdlf_offset, _),
                (rwdlf_found, rwdlf_offset, _),
            ) = self.try_find_line_w_date_for(
                LogFileDateSinceOffsetSeeker.MAX_RWD_FALLBACK_LINES,
                orig_rwdlf_offset,
                orig_rwdlf_offset,
                False,
            )
            log.debug("######### BACKWARDS SEARCH END #########")
        # ... then, forwards.
        if date is None:
            log.debug("######### FORWARDS SEARCH START #########")
            (
                date,
                (fwdlf_found, fwdlf_offset, _),
                (rwdlf_found, rwdlf_offset, _),
            ) = self.try_find_line_w_date_for(
                LogFileDateSinceOffsetSeeker.MAX_FWD_FALLBACK_LINES,
                orig_fwdlf_offset + 1,
                orig_fwdlf_offset,
                True,
            )
            log.debug("######### FORWARDS SEARCH END #########")

        if date:
            self.found_any_date = True
            if date >= self.constraint._since_date:
                self.line_info = (rwdlf_found,
                                  rwdlf_offset,
                                  fwdlf_found,
                                  fwdlf_offset)
        else:
            raise DateSearchFailedAtOffset(
                f"Date search failed at offset `{item}`")

        log.debug(
            "EXTRACTED_DATE: `%s`, SINCE DATE: `%s`, SATISFIES CONDITION?: %s",
            date,
            self.constraint._since_date,
            (date >= self.constraint._since_date) if date else False,
        )
        return date

    def run(self):
        bisect.bisect_left(self, self.constraint._since_date)

        if not self.found_any_date:
            raise NoDateFoundInLogs
        if not self.line_info:
            raise NoLogsFoundSince

        slf_found, slf_offset, elf_found, elf_offset = self.line_info

        # Account for the line feed characters, if found
        line_start = slf_offset + 1 if slf_found else slf_offset
        line_end = elf_offset - 1 if elf_found else elf_offset

        log.debug(
            "RUN END, FOUND LINE(START:%d, END:%d, CONTENT:%s)",
            line_start,
            line_end,
            self.__debug_seek_and_read_line(line_start, line_end)
        )

        return (line_start, line_end)


class SearchConstraintSearchSince(BinarySeekSearchBase):

    def __init__(self, current_date, cache_path, exprs=None,
                 ts_matcher_cls=None, days=0, hours=24, **kwargs):
        """
        A search expression is provided that allows us to identify a datetime
        on each line and check whether it is within a given time period. The
        time period used defaults to 24 hours if use_all_logs is false, 7 days
        if it is true and max_logrotate_depth is default otherwise whatever
        value provided. This can be overridden by providing a specific number
        of hours.

        @param current_date: cli.date(format="+{}".format(self.date_format))
        @param cache_path: path to location where we can create an MPCache
        @param exprs: [DEPRECATED] a list of search/regex expressions used to
                      identify a date/time in. This is deprecated, use
                      ts_matcher_cls.
        @param ts_matcher_cls: TimestampMatcherBase implementation used to
                               match timestamps at start if lines.
        @param days: override default period with number of days
        @param hours: override default period with number of hours
        """
        super().__init__(**kwargs)
        self.cache_path = cache_path
        self.ts_matcher_cls = ts_matcher_cls
        if ts_matcher_cls:
            self.date_format = ts_matcher_cls.DEFAULT_DATETIME_FORMAT
        else:
            log.warning("using patterns to identify timestamp is deprecated - "
                        "use ts_matcher_cls instead")
            self.date_format = TimestampMatcherBase.DEFAULT_DATETIME_FORMAT

        self.current_date = datetime.strptime(current_date, self.date_format)
        self._line_pass = 0
        self._line_fail = 0
        self.exprs = exprs
        self.days = days
        if days:
            self.hours = 0
        else:
            self.hours = hours

        self._results = {}

    def extracted_datetime(self, line):
        if type(line) == bytes:
            # need this for e.g. gzipped files
            line = line.decode("utf-8")

        if self.ts_matcher_cls:
            timestamp = self.ts_matcher_cls(line)
            if timestamp.matched:
                return timestamp.strptime

            return

        # NOTE: the following code can be removed once we remove the deprecated
        # exprs arg from this class.
        log.debug("using patterns to identify timestamp is deprecated - "
                  "use ts_matcher_cls instead")
        for expr in self.exprs:
            # log.debug("attempting to extract from line using expr '%s'",
            #           expr)
            ret = re.search(expr, line)
            if ret:
                # log.debug("expr '%s' successful", expr)
                break

        if not ret:
            # log.info("all exprs unsuccessful: %s", self.exprs)
            return

        str_date = ""
        for g in ret.groups():
            str_date += "{} ".format(g)

        str_date = str_date.strip()
        try:
            return datetime.strptime(str_date, self.date_format)
        except ValueError:
            # this can happen if the line is incomplete or does not contain a
            # timestamp.
            log.exception("")

    @property
    def _is_valid(self):
        return self._since_date is not None

    @cached_property
    def _since_date(self):  # pylint: disable=W0236
        """
        Reflects the date from which we will start to apply searches.
        """
        if not self.current_date:
            return

        return self.current_date - timedelta(days=self.days,
                                             hours=self.hours or 0)

    def apply_to_line(self, line):
        if not self._is_valid:
            log.warning("c:%s unable to apply constraint to line", self.id)
            self._line_pass += 1
            return True

        extracted_datetime = self.extracted_datetime(line)
        if not extracted_datetime:
            self._line_pass += 1
            return True

        ret = self._line_date_is_valid(extracted_datetime)
        if ret:
            self._line_pass += 1
        else:
            self._line_fail += 1

        return ret

    def apply_to_file(self, fd, destructive=True):
        if not self._is_valid:
            log.warning("c:%s unable to apply constraint to %s", self.id,
                        fd.name)
            return

        if fd.name in self._results:
            log.debug("ret cached")
            return self._results[fd.name]

        log.debug("c:%s: starting binary seek search to %s in file %s "
                  "(destructive=True)", self.id, self._since_date, fd.name)
        try:
            seeker = LogFileDateSinceOffsetSeeker(fd, self)
            result = seeker.run()
            fd.seek(result[0] if result and destructive else 0)

            if not result or result[0] == len(seeker):
                self._results[fd.name] = None
            else:
                self._results[fd.name] = result[0]
        except NoDateFoundInLogs:
            log.debug("c:%s No timestamp found in file", self.id)
            fd.seek(0)
            return fd.tell()
        except NoLogsFoundSince:
            log.debug("c:%s No date after found in file", self.id)
            fd.seek(0, 2)
            return fd.tell()
        except DateSearchFailedAtOffset as ed:
            log.debug("c:%s Expanded date search failed for a line: %s",
                      self.id, ed)
            fd.seek(0)
            return fd.tell()

        log.debug("c:%s: finished binary seek search in file %s, offset %d",
                  self.id, fd.name, self._results[fd.name])
        return self._results[fd.name]

    def stats(self):
        _stats = {'line': {'pass': self._line_pass,
                           'fail': self._line_fail}}
        return _stats

    def __repr__(self):
        return ("id={}, since={}, current={}".
                format(self.id, self._since_date, self.current_date))
