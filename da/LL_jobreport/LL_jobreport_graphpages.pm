# Copyright (c) 2023 Forschungszentrum Juelich GmbH.
# This file is part of LLview. 
#
# This is an open source software distributed under the GPLv3 license. More information see the LICENSE file at the top level.
#
# Contributions must follow the Contributor License Agreement. More information see the CONTRIBUTING.md file at the top level.
#
# Contributors:
#    Wolfgang Frings (Forschungszentrum Juelich GmbH) 

package LML_jobreport;

my $VERSION='$Revision: 1.00 $';

use strict;
use Data::Dumper;
use Time::Local;
use Time::HiRes qw ( time );
use JSON;

use lib "$FindBin::RealBin/../lib";
use LML_da_util qw( check_folder );

sub create_graphpages {
  my $self = shift;
  my $DB=shift;

  my $starttime=time();
  my $config_ref=$DB->get_config();

  # 0: init instantiated variables
  ################################
  my $varsetref;
  $varsetref->{"systemname"}=$self->{SYSTEM_NAME};
  if(exists($config_ref->{jobreport}->{paths})) {
    foreach my $p (keys(%{$config_ref->{jobreport}->{paths}})) {
      $varsetref->{$p}=$config_ref->{jobreport}->{paths}->{$p};
    }
  }

  # scan all graphpages
  my $fcount=0;
  foreach my $gpref (@{$config_ref->{jobreport}->{graphpages}}) {
    my $fstarttime=time();
    next if(!exists($gpref->{graphpage}));
    my $fname=$gpref->{graphpage}->{name};
    $fcount++;
    $self->process_graphpage($fname,$gpref->{graphpage},$varsetref);
    printf("%s create_graphpages:[%02d] graphpage %-20s    in %7.4fs\n",$self->{INSTNAME}, $fcount,$fname, time()-$fstarttime);
  }
  
  return();
}


sub process_graphpage {
  my $self = shift;
  my ($fname,$gpref,$varsetref)=@_;
  my ($dsref);
  my $file=$self->apply_varset($gpref->{filepath},$varsetref);

  $dsref=$gpref;

  # get status of datasets from DB
  my $where="name='".$gpref->{name}."'";
  $self->get_datasetstat_from_DB($gpref->{stat_database},$gpref->{stat_table},$where);
  my $ds=$self->{DATASETSTAT}->{$gpref->{stat_database}}->{$gpref->{stat_table}};

  
  # save the JSON file
  my $fh = IO::File->new();
  &check_folder("$file");
  if (!($fh->open("> $file"))) {
    print STDERR "LLmonDB:    WARNING: cannot open $file, skipping...\n";
    return();
  }
  $fh->print($self->encode_JSON($dsref));
  $fh->close();
  # register file
  my $shortfile=$file;$shortfile=~s/$self->{OUTDIR}\///s;
  # update last ts stored to file
  $ds->{$shortfile}->{dataset}=$shortfile;
  $ds->{$shortfile}->{name}=$gpref->{name};
  $ds->{$shortfile}->{ukey}=-1;
  $ds->{$shortfile}->{status}=FSTATUS_EXISTS;
  $ds->{$shortfile}->{checksum}=0;
  $ds->{$shortfile}->{lastts_saved}=$self->{CURRENTTS}; # due to lack of time dependent data
  $ds->{$shortfile}->{mts}=$self->{CURRENTTS}; # last change ts

  # save status of datasets in DB 
  $self->save_datasetstat_in_DB($gpref->{stat_database},$gpref->{stat_table},$where);

}

1;
