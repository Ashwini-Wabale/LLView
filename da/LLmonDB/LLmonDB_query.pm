# Copyright (c) 2023 Forschungszentrum Juelich GmbH.
# This file is part of LLview. 
#
# This is an open source software distributed under the GPLv3 license. More information see the LICENSE file at the top level.
#
# Contributions must follow the Contributor License Agreement. More information see the CONTRIBUTING.md file at the top level.
#
# Contributors:
#    Wolfgang Frings (Forschungszentrum Juelich GmbH) 

package LLmonDB;

my $VERSION='$Revision: 1.00 $';
my $debug=0;

use strict;
use warnings::unused;
use Data::Dumper;
use Time::Local;
use Time::HiRes qw ( time );


sub query {
  my($self) = shift;
  my($db,$table,$optsref)=@_;
  my($resultref);
  printf("  LLmonDB: start query %s/%s\n",$db,$table) if($debug>=3);

  my $dbobj=$self->get_db_handle($db);
  if(!defined($dbobj)) {
    printf(STDERR "[query] \t   LLmonDB_sqlite: ERROR in query, [no db handle] (%s,%s,%s)\n",caller());
    return(undef);
  }
  $resultref=$dbobj->query($table,$optsref);

  printf("  LLmonDB: end query\n") if($debug>=3);

  return($resultref);
}

1;