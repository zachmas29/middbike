class Request:
    """ A single rental request

    Attributes
    ----------
    origin: int
    dest: int
    travel_time: int
    success: bool
    """
    def __init__(self,
                 origin: int = -1,
                 dest: int = -1,
                 minutes_left: int = 0,
                 success: bool = False
                ) -> None:
        self._origin = origin
        self._dest = dest
        self._minutes_left = minutes_left
        self._success = success
    
    @property
    def origin(self) -> int:
        return self._origin
    
    @origin.setter
    def origin(self, value: int) -> None:
        self._origin = value

    @property
    def dest(self) -> int:
        return self._dest
    
    @dest.setter
    def dest(self, value: int) -> None:
        self._dest = value
    
    @property
    def minutes_left(self):
        return self._minutes_left
    
    @minutes_left.setter
    def minutes_left(self, value: int) -> None:
        self._minutes_left
    
    @property
    def success(self) -> bool:
        return self.success
    
    @success.setter
    def success(self, value: bool) -> None:
        return self._success
    
    # stringify
    def __repr__(self) -> str:
        return (f"origin: {self.origin}," 
                f"dest: {self.dest},"
                f"minutes_left: {self.minutes_left},"
                f"success: {self.success}")
