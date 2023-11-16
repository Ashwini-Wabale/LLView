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
my($debug)=0;

use strict;
use Data::Dumper;
use Time::Local;
use Time::Zone;
use Time::HiRes qw ( time );
use Parallel::ForkManager;

use LL_jobreport_datasets_util;
use LL_jobreport_datasets_mngt;
use LL_jobreport_datasets_mngt_solve;
use LL_jobreport_datasets_csv_dat;
use LL_jobreport_datasets_json;
use LL_jobreport_datasets_json_cache;
use LL_jobreport_datasets_access_cache;
use LL_jobreport_datasets_template;
use LL_jobreport_datasets_datatable;

my $TZOFFSET=tz_offset();

sub create_datasets {
  my $self = shift;
  my $DB=shift;
  my $MAX_PROCESSES=shift;

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

  # 1: sort datasets by table_cache and setname (for parallel execution)
  ######################################################################
  my ($sets,@tc_order,$table_cache_first_set);
  
  # scan all datasets in the defintion order
  foreach my $datasetref (@{$config_ref->{jobreport}->{datafiles}}) {
    my $subconfig_ref=$datasetref->{dataset};

    # get setname and tablecache (-none- if not specified)
    my $set="-none-";
    $set=$subconfig_ref->{set} if(exists($subconfig_ref->{set}));
    my $table_cache="-none-";
    if(exists($subconfig_ref->{table_cache})) {
      $table_cache=$subconfig_ref->{table_cache};
      
      # store info about first dataset for this tablecache
      if(!exists($table_cache_first_set->{$table_cache})) {
        $table_cache_first_set->{$table_cache}->{set}=$set; 
        $table_cache_first_set->{$table_cache}->{dataset}=$subconfig_ref;
      }
    }
    # remember order of table caches
    push(@tc_order,$table_cache) if(!exists($sets->{$table_cache}));
    
    # insert dataset in into internal structure
    push(@{ $sets->{$table_cache}->{$set} },$subconfig_ref);
  }

  # 2: create order of parallel execution
  #######################################
  my $actions; 
  
  #  1st schedule all datasets of sets without a table_cache
  foreach my $set (keys(%{$sets->{"-none-"}})) {
    my $action;
    $action->{type}="fork";	$action->{table_cache}="-none-"; $action->{set}=$set;
    push(@{$actions},$action);
  }

  #  2nd schedule all datasets of sets with table_cache
  foreach my $table_cache (@tc_order) {
    next if($table_cache eq "-none-");

    #  cache table in PRIMARY process
    {
      my $action;
      $action->{type}="cache_table";	$action->{table_cache}=$table_cache;
      $action->{set}=$table_cache_first_set->{$table_cache}->{set};
      $action->{format}=$table_cache_first_set->{$table_cache}->{dataset}->{format};
      push(@{$actions},$action);
    }
    
    #  run scan of sets which belongs to that table_cache in PRIMARY process
    foreach my $set (keys(%{$sets->{$table_cache}})) {
      {
        my $action;
        $action->{type}="scan_table";	$action->{table_cache}=$table_cache; $action->{set}=$set;
        $action->{format}=$table_cache_first_set->{$table_cache}->{dataset}->{format};
        push(@{$actions},$action);
      }
    }
    
    #  fork parallel file write operations
    my $parlevel=12;
    $parlevel=$table_cache_first_set->{$table_cache}->{dataset}->{table_cache_par_level}
              if(exists($table_cache_first_set->{$table_cache}->{dataset}->{table_cache_par_level}));
    for(my $count=0;$count<$parlevel;$count++) {
      my $action;
      $action->{type}="fileio";  $action->{table_cache}=$table_cache; $action->{set}="";
      $action->{part}=$count;    $action->{parlevel}=$parlevel;
      $action->{format}=$table_cache_first_set->{$table_cache}->{dataset}->{format};
      push(@{$actions},$action);
    }

    #  drop cached table in PRIMARY process
    {
      my $action;
      $action->{type}="drop_table";	$action->{table_cache}=$table_cache;
      $action->{set}=$table_cache_first_set->{$table_cache}->{set};
      $action->{format}=$table_cache_first_set->{$table_cache}->{dataset}->{format};
      push(@{$actions},$action);
    }
  }

  # DEBUG report on actions
  if($debug) {
    printf("%s create_datasets: init in %7.4fs\n",$self->{INSTNAME}, time()-$starttime);
    printf("%s create_datasets: order -->\n",$self->{INSTNAME});
    my $action_count=0;
    foreach my $action (@{$actions}) {
      my @dsnames;
      foreach my $dataset (@{$sets->{$action->{table_cache}}->{$action->{set}}}) {
        push(@dsnames,$dataset->{name});
      }
      printf("%s   %02d: %-12s -> %-12s %-20s [%s] [%s of %s] format=%s\n",$self->{INSTNAME},++$action_count,$action->{type},
              $action->{table_cache},$action->{set},
              join(",",@dsnames),
              exists($action->{part})?$action->{part}:"-",	
              exists($action->{parlevel})?$action->{parlevel}:"-",
              exists($action->{format})?$action->{format}:"-"
            );
    }
  }

  # 3: start PARALLEL Processing
  ##############################
  my $pm = Parallel::ForkManager->new($MAX_PROCESSES);

  printf("%s create_datasets: start parallel execution MAX_PROCESSES=%d\n",$self->{INSTNAME},$MAX_PROCESSES);
  
  my $parstarttime=time();
  my $action_count=0;
  DATA_LOOP:
  foreach my $action (@{$actions}) {
    $action_count++;

    # dataset for this action
    my @dsnames;
    foreach my $dataset (@{$sets->{$action->{table_cache}}->{$action->{set}}}) {
      push(@dsnames,$dataset->{name});
    }
    printf("%s S%03d [at %7.4fs] perform action %-12s -> %-12s %-20s [%s] [%s of %s] \n",$self->{INSTNAME},$action_count,
            time()-$parstarttime,
            $action->{type},$action->{table_cache},$action->{set}, join(",",@dsnames),
            exists($action->{part})?$action->{part}:"-",	
            exists($action->{parlevel})?$action->{parlevel}:"-");
    
    if($action->{type} eq "cache_table") {
      # cache tables in PRIMARY process
      if($action->{format} eq "json") {
        $self->process_data_query_cache_table_json( $action->{table_cache},
                                                    $table_cache_first_set->{$action->{table_cache}}->{dataset},
                                                    $varsetref);
      }
      if($action->{format} eq "access") {
        $self->process_data_query_cache_table_access( $action->{table_cache},
                                                      $table_cache_first_set->{$action->{table_cache}}->{dataset},
                                                      $varsetref);
      }
      next DATA_LOOP;
    } 
    if($action->{type} eq "scan_table") {
      # scan tables in PRIMARY process
      foreach my $dataset (@{$sets->{$action->{table_cache}}->{$action->{set}}}) {
        if($dataset->{format}=~/^(json)$/) {
          $self->process_dataset_json($DB,$dataset,$varsetref);
        }
        if($dataset->{format}=~/^(access)$/) {
          $self->process_dataset_access($DB,$dataset,$varsetref);
        }
      }
      next DATA_LOOP;
    }
    if($action->{type} eq "drop_table") {
      # cache tables in PRIMARY process
      if($action->{format} eq "json") {
        $self->process_data_query_drop_table_json($action->{table_cache},
                                                  $table_cache_first_set->{$action->{table_cache}}->{dataset},
                                                  $varsetref);
      }
      if($action->{format} eq "access") {
        $self->process_data_query_drop_table_access($action->{table_cache},
                                                    $table_cache_first_set->{$action->{table_cache}}->{dataset},
                                                    $varsetref);
      }
      next DATA_LOOP;
    } 
    
    my $set=$action->{set};

    # PARALLEL: 
    #   Forks and returns the pid for the child:
    my $pid = $pm->start and next DATA_LOOP;

    my $startsetts=time();

    if($action->{type} eq "fileio") {
      my $caller=$0; $caller=~s/^.*\/([^\/]+)$/$1/gs; # set INSTNAME for messages
      $self->{INSTNAME}=sprintf("[%s][FILEIO%02d_%s]",$caller,$action->{part},substr($action->{table_cache},0,7));
      
      # Parallel FILEIO operation for datasets using tablecache
      if($action->{format} eq "json") {
        $self->write_data_to_file_json_cache($action->{table_cache},$action->{part},$action->{parlevel});
      } 
      if($action->{format} eq "access") {
        $self->write_data_to_file_access_cache($action->{table_cache},$action->{part},$action->{parlevel});
      }
      my $endtime=time();
      printf("%s S%03d [at %7.4fs] FINISHED process_dataset: %-20s in %7.4fs (ts=%.5f,%.5f,l=%d,nr=%d)\n",
              $self->{INSTNAME},$action_count,$endtime-$parstarttime,
              $action->{table_cache}, $endtime-$startsetts,$startsetts,$endtime,2,$action_count);
    } else {
      my $caller=$0;$caller=~s/^.*\/([^\/]+)$/$1/gs; # set INSTNAME for messages
      $self->{INSTNAME}=sprintf("[%s][%s]",$caller,substr(uc($set),0,16));

      # Parallel operation for other datasets
      foreach my $dataset (@{$sets->{$action->{table_cache}}->{$action->{set}}}) {
        printf("%s create_datasets: start work on %s\n",$self->{INSTNAME},$dataset->{name});
        if($dataset->{format}=~/^(dat|csv)$/) {
          $self->process_dataset_cvs_dat($DB,$dataset,$varsetref);
        } elsif($dataset->{format} eq "json") {
          $self->process_dataset_json($DB,$dataset,$varsetref);
        } elsif($dataset->{format} eq "registerfile") {
          $self->process_dataset_register($DB,$dataset,$varsetref);
        } elsif($dataset->{format} eq "template") {
          $self->process_dataset_template($DB,$dataset,$varsetref);
        } elsif($dataset->{format} eq "datatable") {
          $self->process_dataset_datatable($DB,$dataset,$varsetref);
        } elsif($dataset->{format} eq "access") {
          $self->process_dataset_access($DB,$dataset,$varsetref);
        } else {
          printf("%s process_dataset:      unknown file format %s\n",$self->{INSTNAME},$dataset->{format});
        }
        # printf("%s create_datasets: end work on %s\n",$self->{INSTNAME},$dataset->{name});
      }
      my $endtime=time();
      printf("%s S%03d [at %7.4fs] FINISHED process_dataset: %-20s in %7.4fs (ts=%.5f,%.5f,l=%d,nr=%d)\n",
              $self->{INSTNAME},$action_count,$endtime-$parstarttime,
              $set, $endtime-$startsetts,$startsetts,$endtime,2,$action_count);
    }
    $pm->finish; # Terminates the child process
  }
  printf("%s create_datasets: wait for childs (%d),       after %7.4ss\n",$self->{INSTNAME}, scalar $pm->running_procs(),time()-$parstarttime);
  $pm->wait_all_children;
  printf("%s create_datasets: parallel ready steps           in %7.4fs\n",$self->{INSTNAME}, time()-$parstarttime);

  return();
}

sub process_dataset_json {
  my $self = shift;
  my($DB,$dataset,$varsetref)=@_;
  my $starttime=time();
  
  # printf("%s process_dataset_json:    start work on %s\n",$self->{INSTNAME},$dataset->{name});

  $self->parse_filemap($dataset);
  
  # get status of datasets from DB
  $self->get_datasetstat_from_DB($dataset->{stat_database},$dataset->{stat_table});

  # RUN: query for new data and store in file
  ###############################################
  $self->{COUNT_OP_NEW_FILE}=0;
  $self->{COUNT_OP_EXISTING_FILE}=0;
  $self->{COUNT_OP_WRITE_LINE}=0;

  if(!exists($dataset->{table_cache})) {
    if(exists($dataset->{FORALL})) {
      if(exists($dataset->{column_filemap})) {
        # process data in own query and splitting afterwards data to files
        $self->process_FORALL($DB,$dataset,$varsetref,\&check_filemapping,$dataset);
        $self->process_data_query_and_save_json($dataset,$varsetref);
      } else {
        # run a query for each value of FORALL variable
        $self->process_FORALL($DB,$dataset,$varsetref,\&process_data_query_and_save_json,$dataset);
      }
    } else {
      # only one query (no cache required)
      $self->process_data_query_and_save_json($dataset,$varsetref);
    }
  } else {
    if(exists($dataset->{FORALL})) {
      if(exists($dataset->{column_filemap})) {
        # process data in own query and splitting afterwards data to files
        $self->process_FORALL($DB,$dataset,$varsetref,\&check_filemapping,$dataset);
        $self->process_data_query_and_save_json_cache($dataset,$varsetref);
      } else {
        # preset info about files in ds structure
        $self->process_FORALL($DB,$dataset,$varsetref,\&check_filemapping,$dataset);
        # run a query for each value of FORALL variable
        $self->process_FORALL($DB,$dataset,$varsetref,\&process_data_query_and_save_json_cache,$dataset);
      }
    } else {
      # preset info about file in ds structure
      $self->check_filemapping($dataset,$varsetref);
      # only one query (no cache required)
      $self->process_data_query_and_save_json_cache($dataset,$varsetref);
    }
  }

  # save status of datasets in DB 
  $self->save_datasetstat_in_DB($dataset->{stat_database},$dataset->{stat_table});

  printf("%s process_dataset_json:     end work (#files created: %4d, #files appended: %4d: #lines=%6d) in %7.4fs on %-25s\n",$self->{INSTNAME},
          $self->{COUNT_OP_NEW_FILE},
          $self->{COUNT_OP_EXISTING_FILE},
          $self->{COUNT_OP_WRITE_LINE},
          time()-$starttime,$dataset->{name});
}

sub process_dataset_access {
  my $self = shift;
  my($DB,$dataset,$varsetref)=@_;
  my $starttime=time();
  
  # printf("%s process_dataset_access:    start work on %s\n",$self->{INSTNAME},$dataset->{name});

  $self->parse_filemap($dataset);
  
  # get status of datasets from DB
  $self->get_datasetstat_from_DB($dataset->{stat_database},$dataset->{stat_table});

  # RUN: query for new data and store in file
  ###############################################
  $self->{COUNT_OP_NEW_FILE}=0;
  $self->{COUNT_OP_EXISTING_FILE}=0;
  $self->{COUNT_OP_WRITE_LINE}=0;
  
  if(!exists($dataset->{table_cache})) {
    if(exists($dataset->{FORALL})) {
      if(exists($dataset->{column_filemap})) {
        # process data in own query and splitting afterwards data to files
        # $self->process_FORALL($DB,$dataset,$varsetref,\&check_filemapping,$dataset);
        # $self->process_data_query_and_save_access($dataset,$varsetref);
      } else {
        # run a query for each value of FORALL variable
        # $self->process_FORALL($DB,$dataset,$varsetref,\&process_data_query_and_save_access,$dataset);
      }
  } else {
    # preset info about file in ds structure
    $self->check_filemapping($dataset,$varsetref);
    # only one query (no cache required)
    $self->process_data_query_and_save_access($dataset,$varsetref);
  }
  } else {
    if(exists($dataset->{FORALL})) {
      if(exists($dataset->{column_filemap})) {
        # process data in own query and splitting afterwards data to files
        $self->process_FORALL($DB,$dataset,$varsetref,\&check_filemapping,$dataset);
        $self->process_data_query_and_save_access_cache($dataset,$varsetref);
      } else {
        # preset info about files in ds structure
        $self->process_FORALL($DB,$dataset,$varsetref,\&check_filemapping,$dataset);
        # run a query for each value of FORALL variable
        $self->process_FORALL($DB,$dataset,$varsetref,\&process_data_query_and_save_access_cache,$dataset);
      }
    } else {
      # preset info about file in ds structure
      $self->check_filemapping($dataset,$varsetref);
      # only one query (no cache required)
      $self->process_data_query_and_save_access_cache($dataset,$varsetref);
    }
  }

  # save status of datasets in DB 
  $self->save_datasetstat_in_DB($dataset->{stat_database},$dataset->{stat_table});

  printf("%s process_dataset_access:     end work (#files created: %4d, #files appended: %4d: #lines=%6d) in %7.4fs on %-25s\n",$self->{INSTNAME},
          $self->{COUNT_OP_NEW_FILE},
          $self->{COUNT_OP_EXISTING_FILE},
          $self->{COUNT_OP_WRITE_LINE},
          time()-$starttime,$dataset->{name}) if($debug);
}

sub process_dataset_cvs_dat {
  my $self = shift;
  my ($DB,$dataset,$varsetref)=@_;
  my $starttime=time();
  my $filepath_parsed=undef; 	# required for signle output file without colmap
  
  printf("%s process_dataset:    start work on %s\n",$self->{INSTNAME},$dataset->{name});

  my $starttime1=time();
  $self->parse_filemap($dataset);
  printf("%s process_dataset:    first run (parse ) after %7.4fs on %s\n",$self->{INSTNAME},time()-$starttime1,$dataset->{name});

  # get status of datasets from DB
  $starttime1=time();
  my $where="dataset not like '%unknown%'"; # get rid of old entries with unknown info (datasets)
  $self->get_datasetstat_from_DB($dataset->{stat_database},$dataset->{stat_table},$where);
  printf("%s process_dataset:    first run (get_ds ) after %7.4fs on %s\n",$self->{INSTNAME},time()-$starttime1,$dataset->{name});

  # 1st RUN: update state in DB for each new file
  ###############################################
  $starttime1=time();
  if(exists($dataset->{FORALL})) {
    $self->process_FORALL($DB,$dataset,$varsetref,\&check_filepath,$dataset);
  } else {
    $filepath_parsed=$self->check_filepath($dataset,$varsetref);
  }
  printf("%s process_dataset:    first run (checkfp) after %7.4fs on %s\n",$self->{INSTNAME},time()-$starttime1,$dataset->{name});
  # save status of datasets in DB (required by following query)
  $starttime1=time();
  $self->save_datasetstat_in_DB($dataset->{stat_database},$dataset->{stat_table},$where);
  printf("%s process_dataset:    first run (save_ds) after %7.4fs on %s (%s,%s)\n",$self->{INSTNAME},time()-$starttime1,$dataset->{name},$dataset->{stat_database},$dataset->{stat_table});
  printf("%s process_dataset:    finished first  run after %7.4fs on %s\n",$self->{INSTNAME},time()-$starttime,$dataset->{name});

  # 2nd RUN: query for new data and store in file
  ###############################################
  my $starttime2=time();
  $self->{DB}->close_db($dataset->{stat_database});

  $self->{COUNT_OP_NEW_FILE}=0;
  $self->{COUNT_OP_EXISTING_FILE}=0;
  $self->{COUNT_OP_WRITE_LINE}=0;

  $self->process_data_query_and_save_csv_dat($dataset,$filepath_parsed);
  printf("%s process_dataset:    finished second run after %7.4fs on %s\n",$self->{INSTNAME},time()-$starttime2,$dataset->{name});

  # save status of datasets in DB again (ts updated)
  $self->save_datasetstat_in_DB($dataset->{stat_database},$dataset->{stat_table},$where);

  printf("%s process_dataset_cvs_dat:  end work (#files created: %4d, #files appended: %4d: #lines=%6d) in %7.4fs on %-25s\n",$self->{INSTNAME},
          $self->{COUNT_OP_NEW_FILE},
          $self->{COUNT_OP_EXISTING_FILE},
          $self->{COUNT_OP_WRITE_LINE},
          time()-$starttime,$dataset->{name});
}

sub process_dataset_register {
  my $self = shift;
  my ($DB,$dataset,$varsetref)=@_;
  my $starttime=time();
  
  # printf("%s process_dataset_register:    start work on %s\n",$self->{INSTNAME},$dataset->{name});

  $self->parse_filemap($dataset);
  
  # get status of datasets from DB
  $self->get_datasetstat_from_DB($dataset->{stat_database},$dataset->{stat_table});

  # RUN: query for new data and store in file
  ###############################################
  $self->{COUNT_OP_NEW_FILE}=0;
  $self->{COUNT_OP_EXISTING_FILE}=0;
  $self->{COUNT_OP_WRITE_LINE}=0;

  if(exists($dataset->{FORALL})) {
    if(exists($dataset->{column_filemap})) {
      # process data in own query and splitting afterwards data to files
      $self->process_FORALL($DB,$dataset,$varsetref,\&check_fileregister,$dataset);
    }
  } else {
    # only one query 
    $self->process_data_query_and_save_json($dataset,$varsetref);
  }

  # save status of datasets in DB 
  $self->save_datasetstat_in_DB($dataset->{stat_database},$dataset->{stat_table});

  printf("%s process_dataset_register: end work (#files reg.:    %4d, #files updated:  %4d: #lines=%6d) in %7.4fs on %-25s\n",$self->{INSTNAME},
          $self->{COUNT_OP_NEW_FILE},
          $self->{COUNT_OP_EXISTING_FILE},
          $self->{COUNT_OP_WRITE_LINE},
          time()-$starttime,$dataset->{name});
}


sub parse_filemap {
  my $self = shift;
  my ($dataset)=@_;
  
  # how data is mapped to output files
  if(exists($dataset->{column_filemap})) {
    foreach my $pair (split(/\s*,\s*/,$dataset->{column_filemap})) {
      my($v,$col)=split(":",$pair);
      # main map var
      $self->{DATASETSTAT_MAP}->{$dataset->{name}}->{filemap_v}=$v;
      $self->{DATASETSTAT_MAP}->{$dataset->{name}}->{filemap_col}=$col;
      # list of all map vars
      push(@{$self->{DATASETSTAT_MAP}->{$dataset->{name}}->{v_list}},$v);
      push(@{$self->{DATASETSTAT_MAP}->{$dataset->{name}}->{col_list}},$col);
      $self->{DATASETSTAT_MAP}->{$dataset->{name}}->{all_v_to_col}->{$v}=$col;
      $self->{DATASETSTAT_MAP}->{$dataset->{name}}->{all_col_to_v}->{$col}=$v;
    }
    $self->{DATASETSTAT_MAP}->{$dataset->{name}}->{column_ts}=$dataset->{column_ts};
  }

  if(exists($dataset->{filemap})) {
    my($v)=$dataset->{filemap};
    $self->{DATASETSTAT_MAP}->{$dataset->{name}}->{filemap_v}=$v;
  } 
  # print "parse_filemap: $dataset->{name}:", Dumper($self->{DATASETSTAT_MAP});
}

sub check_filepath {
  my $self = shift;
  my($dataset,$varsetref)=@_;
  my $ds=$self->{DATASETSTAT}->{$dataset->{stat_database}}->{$dataset->{stat_table}};
  
  # printf("%s check_filepath:       start work on %s\n",$self->{INSTNAME},$dataset->{filepath});
  my $file=$dataset->{filepath};
  
  #  replace vars
  while ( my ($key, $value) = each(%{$varsetref}) ) {
    $file=~s/\$\{$key\}/$value/gs;
    $file=~s/\$$key/$value/gs;
    $value=~s/\//:/gs;  # replace '/' in vars which build a file path
    $file=~s/\$\{\{$key\}\}/$value/gs;
  }
  my $shortfile=$file;$shortfile=~s/$self->{OUTDIR}\///s;

  printf("%s check_filepath:         WARNING %s\n",$self->{INSTNAME},$shortfile) if($shortfile=~/unknown/);

  my $init_file=0;
  if(!exists($ds->{$shortfile})) {
    $init_file=1;
  } elsif(exists($dataset->{renew_daily})) {
    if($dataset->{renew_daily}=~/^(1|yes)$/) {
      my $window_start=1*3600+42*60;
      my $window_end=$window_start-15*60;
      my $ts_today=$self->{CURRENTTS}-$self->{CURRENTTS}%(24*3600)-$TZOFFSET; # start of day
      my $now_ts_today=$self->{CURRENTTS}%(24*3600); # seconds today
      my $lastsaved_ts_today=$ds->{$shortfile}->{lastts_saved}-$ts_today;
      # printf("%s check_filepath: for renew file:check %s ts_today=%d now_ts_today=%d lastsaved_ts_today=%d window: %d..%d\n",$self->{INSTNAME},
      #         $shortfile,$ts_today,$now_ts_today,$lastsaved_ts_today,$window_start,$window_end );
      if(($now_ts_today>=$window_start) && ($now_ts_today<=$window_end)) {
        if($lastsaved_ts_today<$window_start) {
          $init_file=1;
          printf("%s check_filepath: renew file, DO    %s ts_today=%d now_ts_today=%d lastsaved_ts_today=%d\n",$self->{INSTNAME},
                  $shortfile,$ts_today,$now_ts_today,$lastsaved_ts_today );
        }
      }
    }
  } elsif(exists($dataset->{renew})) {
    if($dataset->{renew}=~/daily/) {
      my $window_start=15*3600+42*60;
      if($dataset->{renew}=~/daily\((\d\d)\:(\d\d)\)/) {
        $window_start=$1*3600+$2*60;
        # printf("%s check_filepath: found renew_daily %s,%s -> %d (%s) tzoffet=%d\n",$self->{INSTNAME},$1,$2,$window_start,$shortfile,$TZOFFSET);
      }
      my $window_end=$window_start+15*60;
      my $ts_today=$self->{CURRENTTS}-$self->{CURRENTTS}%(24*3600)-$TZOFFSET; # start of day
      my $now_ts_today=$self->{CURRENTTS}%(24*3600)+$TZOFFSET; # seconds today
      my $lastsaved_ts_today=$ds->{$shortfile}->{lastts_saved}-$ts_today;
      
      # printf("%s check_filepath: for renew file:check %s ts_today=%d now_ts_today=%d lastsaved_ts_today=%d window: %d..%d\n",$self->{INSTNAME},
      #         $shortfile,$ts_today,$now_ts_today,$lastsaved_ts_today,$window_start,$window_end );
      if(($now_ts_today>=$window_start) && ($now_ts_today<=$window_end)) {
        if($lastsaved_ts_today<$window_start) {
          $init_file=1;
          printf("%s check_filepath: renew file, DO    %s ts_today=%d now_ts_today=%d lastsaved_ts_today=%d\n",$self->{INSTNAME},
                  $shortfile,$ts_today,$now_ts_today,$lastsaved_ts_today );
        }
      }
    } 
    if($dataset->{renew}=~/always/) {
      $init_file=1;
      # printf("%s check_filepath: renew file, ALWAYS  %s \n",$self->{INSTNAME}, $shortfile );
    }
    if($dataset->{renew}=~/delta/) {
      $ds->{$shortfile}->{status}=0;
      printf("%s: check_filepath: renew file, DELTA  %s \n",$self->{INSTNAME}, $shortfile );
    }
  }
  
  # check if new file
  if($init_file) {
    $ds->{$shortfile}->{dataset}=$shortfile;
    $ds->{$shortfile}->{name}=$dataset->{name};
    
    if(exists($ds->{$shortfile}->{lastts_saved})) {
      $ds->{$shortfile}->{lastts_saved}=1; # its a file to be renewed
    } else {
      # its a new new file which may be not written (if there is no data)
      $ds->{$shortfile}->{lastts_saved}=int($self->{CURRENTTS}-365*24*3600); # mark time when this entry was created minus delay
    }
    $ds->{$shortfile}->{checksum}=0;
    # internal mapping
    if(exists($self->{DATASETSTAT_MAP}->{$dataset->{name}})) {
      my $v=$self->{DATASETSTAT_MAP}->{$dataset->{name}}->{filemap_v};
      if(defined($v)) {
        $ds->{$shortfile}->{ukey}=$varsetref->{$v};
      } else {
        $ds->{$shortfile}->{ukey}=-1;
      }
    } else {
      $ds->{$shortfile}->{ukey}=-1;
    }
    $ds->{$shortfile}->{status}=0;
    printf("%s check_filepath:         init ds for %s\n",$self->{INSTNAME},$file) if($file=~/fabric/); # if($debug);
  }
  # printf("%s check_filepath:       end   work on %s\n",$self->{INSTNAME},$dataset->{filepath});
  return($shortfile);
}

sub check_filemapping {
  my $self = shift;
  my ($dataset,$varsetref)=@_;
  my $ds=$self->{DATASETSTAT}->{$dataset->{stat_database}}->{$dataset->{stat_table}};
  my $file=$dataset->{filepath};

  #  replace vars in filepath
  while ( my ($key, $value) = each(%{$varsetref}) ) {
    $file=~s/\$\{$key\}/$value/gs;
    $file=~s/\$$key/$value/gs;
    $value=~s/\//:/gs;  # replace '/' in vars which build a file path
    $file=~s/\$\{\{$key\}\}/$value/gs;
  }

  my $shortfile=$file;$shortfile=~s/$self->{OUTDIR}\///s;
  
  if(!exists($ds->{$shortfile})) {
    $ds->{$shortfile}->{dataset}=$shortfile;
    $ds->{$shortfile}->{name}=$dataset->{name};
    $ds->{$shortfile}->{lastts_saved}=int($self->{CURRENTTS}); # mark time when this entry was created	
    $ds->{$shortfile}->{checksum}=0;
    # internal mapping
    if(exists($self->{DATASETSTAT_MAP}->{$dataset->{name}})) {
      my $v=$self->{DATASETSTAT_MAP}->{$dataset->{name}}->{filemap_v};
      if(defined($v)) {
        $ds->{$shortfile}->{ukey}=$varsetref->{$v};
      } else {
        $ds->{$shortfile}->{ukey}=-1;
      }
    } else {
      $ds->{$shortfile}->{ukey}=-1;
    }
    $ds->{$shortfile}->{status}=0;
    #	printf("%s check_filepath:         init ds for %s\n",$self->{INSTNAME},$file) if($debug);
  }

  # build search key
  my @skeys;
  foreach my $v (@{$self->{DATASETSTAT_MAP}->{$dataset->{name}}->{v_list}}) {
    push(@skeys,$varsetref->{$v});
  }
  my $skey=join(":",@skeys);
  $self->{DATASETSTAT_MAP}->{$dataset->{name}}->{skey_to_filename}->{$skey}=$shortfile;
  
  # printf("%s check_filepath:       skey=%-25s file=%s\n",$self->{INSTNAME},$skey,$file);
}

sub check_fileregister {
  my $self = shift;
  my ($dataset,$varsetref)=@_;
  my $ds=$self->{DATASETSTAT}->{$dataset->{stat_database}}->{$dataset->{stat_table}};
  
  # printf("%s check_fileregister:       start work on %s\n",$self->{INSTNAME},$dataset->{filepath});
  my $file=$dataset->{filepath};
  #  replace vars
  while ( my ($key, $value) = each(%{$varsetref}) ) {
    $file=~s/\$\{$key\}/$value/gs;
    $file=~s/\$$key/$value/gs;
    $value=~s/\//:/gs;  # replace '/' in vars which build a file path
    $file=~s/\$\{\{$key\}\}/$value/gs;
  }
  my $shortfile=$file;$shortfile=~s/$self->{OUTDIR}\///s;
  # printf("%s check_fileregister:       file=%s -> %s\n",$self->{INSTNAME},$file,$shortfile);
  # check if new file
  if(!exists($ds->{$shortfile})) {
    $ds->{$shortfile}->{dataset}=$shortfile;
    $ds->{$shortfile}->{name}=$dataset->{name};
    $ds->{$shortfile}->{lastts_saved}=int($self->{CURRENTTS}); # mark time when this entry was created
    $ds->{$shortfile}->{checksum}=0;
    # internal mapping
    if(exists($self->{DATASETSTAT_MAP}->{$dataset->{name}})) {
      my $v=$self->{DATASETSTAT_MAP}->{$dataset->{name}}->{filemap_v};
      if(defined($v)) {
        $ds->{$shortfile}->{ukey}=$varsetref->{$v};
      } else {
        $ds->{$shortfile}->{ukey}=-1;
      }
    } else {
      $ds->{$shortfile}->{ukey}=-1;
    }
  }
  if( -f $file ) {
    my ($dev,$ino,$mode,$nlink,$uid,$gid,$rdev,$size, $atime,$mtime,$ctime,$blksize,$blocks) = stat($file);
    $ds->{$shortfile}->{status}=1;
    $ds->{$shortfile}->{lastts_saved}=$mtime;
    $ds->{$shortfile}->{checksum}=0;
  } else {
    $ds->{$shortfile}->{status}=0;
  }
  # printf("%s check_fileregister:       end   work on %s\n",$self->{INSTNAME},$dataset->{filepath});
  return($shortfile);
}

1;
