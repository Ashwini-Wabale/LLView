#!/usr/bin/perl -w
# Copyright (c) 2023 Forschungszentrum Juelich GmbH.
# This file is part of LLview. 
#
# This is an open source software distributed under the GPLv3 license. More information see the LICENSE file at the top level.
#
# Contributions must follow the Contributor License Agreement. More information see the CONTRIBUTING.md file at the top level.
#
# Contributors:
#    Wolfgang Frings (Forschungszentrum Juelich GmbH) 
#    Filipe Guimarães (Forschungszentrum Juelich GmbH) 
#
# This script runs a command that will be executed only every <niter> call of this script.
# Optionally, if specified, an empty input file is copied from 'empty_from' to 'empty_to'
use strict;
use Getopt::Long;

my $opt_verbose=0;
my $opt_stepfile = "stepfile.dat";
my $opt_niter    = 15;
my $opt_emptyfilefrom = undef;
my $opt_emptyfileto   = undef;

usage($0) if( ! GetOptions( 
                'verbose'            => \$opt_verbose,
                'stepfile=s'         => \$opt_stepfile,
                'niter=i'            => \$opt_niter,
                'empty_from=s'       => \$opt_emptyfilefrom,
                'empty_to=s'         => \$opt_emptyfileto
                ) );

my $command = join(" ",@ARGV);

printf("%s: stepfile = %s\n",$0,$opt_stepfile);
printf("%s: niter    = %d\n",$0,$opt_niter);
printf("%s: empty    = %s -> %s\n",$0,$opt_emptyfilefrom,$opt_emptyfileto) if(defined($opt_emptyfilefrom) && defined($opt_emptyfileto));
printf("%s: command  = %s\n",$0,$command);

my $current_step=0;

if (-f $opt_stepfile ) {
  $current_step=`cat $opt_stepfile`;
}
# Adding current step and restarting count if needed
$current_step = ++$current_step % $opt_niter;
open(STEP, "> $opt_stepfile") or die "cannot open $opt_stepfile";
print STEP $current_step;
close(STEP);

printf("%s: current  = %d of %d\n",$0,$current_step % $opt_niter,$opt_niter);

my $rc=0;
if($current_step % $opt_niter != 0) {
  if(defined($opt_emptyfilefrom) && defined($opt_emptyfileto)) {
    my $cmd="cp $opt_emptyfilefrom $opt_emptyfileto";
    system($cmd); $rc=$?;
    if($rc) {
      printf STDERR "failed executing: %s rc=%d\n",$cmd,$rc; exit(-1);
    }
  } else {
    printf("%s: --> skip execution\n",$0);
  }
} else {
  printf("%s: executing: %s\n",$0,$command);
  system($command); $rc=$?;
  if($rc) {
    printf STDERR "failed executing: %s rc=%d\n",$command,$rc; exit(-1);
  }
}

exit;


sub usage {
  die "Usage: $_[0] <options> [--] <commands>
                -stepfile <file>           : file containing internal counter
                -niter <num>               : skip <niter>-1 steps
                -empty_from <file>         : template file for generation empty input file
                -empty_to <file>           : destination path/fname for empty input file
                -verbose                   : verbose

        The command will be executed only every <niter> call of this script.
        Optionally, if specified, an empty input file will be generated.
";
}

