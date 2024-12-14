package LML_jobreport;

my $VERSION='$Revision: 1.00 $';
my($debug)=0;

use strict;
use constant  {
    FSTATUS_UNKNOWN => -1,
    FSTATUS_NOT_EXISTS => 0,
    FSTATUS_EXISTS => 1,
    FSTATUS_COMPRESSED => 2,
    FSTATUS_TOBEDELETED => 3,
    FSTATUS_DELETED => 4,
    FSTATUS_TOBECOMPRESSED => 5,
    FACTION_COMPRESS => 1,
    FACTION_ARCHIVE => 2,
    FACTION_REMOVE => 3
};

1;
