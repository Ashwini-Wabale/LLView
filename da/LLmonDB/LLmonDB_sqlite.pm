# Copyright (c) 2023 Forschungszentrum Juelich GmbH.
# This file is part of LLview. 
#
# This is an open source software distributed under the GPLv3 license. More information see the LICENSE file at the top level.
#
# Contributions must follow the Contributor License Agreement. More information see the CONTRIBUTING.md file at the top level.
#
# Contributors:
#    Wolfgang Frings (Forschungszentrum Juelich GmbH) 

package LLmonDB_sqlite;

my $VERSION='$Revision: 1.00 $';
my $debug=0;
my($LOGACCESS)=0;

use strict;
use warnings::unused;
use Data::Dumper;
use Time::Local;
use Time::HiRes qw ( time );
use DBI;
use DBD::SQLite;

sub new {
  my $self    = {};
  my $proto   = shift;
  my $class   = ref($proto) || $proto;
  my $dbdir   = shift;
  my $dbname  = shift;
  my $verbose = shift;

  printf("  LLmonDB_sqlite: new %s\n",ref($proto)) if($debug>=3);
  $self->{VERBOSE}   = $verbose; 
  $self->{DBINIT}    = 0; 
  $self->{DBDIR}     = $dbdir; 
  $self->{DBNAME}    = $dbname; 
  $self->{FNAME}     = "$dbdir/LLmonDB_${dbname}.sqlite"; 
  $self->{INSTNAME}  = $0; $self->{INSTNAME}=~s/^.*\///gs; $self->{INSTNAME}="[$self->{INSTNAME}]";
  $self->{EOTS}      = 0;
  $self->{AUTOCOMMIT} = 1; 

  bless $self, $class;
  return $self;
}

sub LOGREPORT {
  my $self = shift;
  my($dbname,$op,$caller1,$caller2,$caller3,$err)=@_;
  return() if(!$LOGACCESS);
  my $ts=time();

  my $logname = "$self->{DBDIR}/log/LLmonDB_${dbname}.log"; 
  open(LOG, ">> $logname");
  my $str=sprintf("[%10d]%10d+%4d: %-20s %-15s (SCRIPT:%s) (ERR:%s) (CALLER:%s,%s,%s)\n",$$,
                    $ts,
                    ($self->{EOTS}>0)?$ts-$self->{EOTS}:0,
                    $dbname,
                    $op,
                    $self->{INSTNAME},
                    defined($err)?$err:"-",
                    $caller1,$caller2,$caller3);
  
  print LOG $str;
  close(LOG);
}

sub mycommit() {
  my $self = shift;
  
  if(!$self->{AUTOCOMMIT}) {
    $self->{DBH}->commit() or die $self->{DBH}->errstr;
  }
}


sub DESTROY {
  my $self = shift;

  if($self->{DBINIT}) {
    $self->LOGREPORT($self->{DBNAME},"destroy without close",caller(),"");
  }
  
  # disconnect from the database
  $self->close_db();
}

# check on existence of data base file 
sub check_db_file {
  my($self) = shift;

  if(-f $self->{FNAME}) {
    return(1);
  } else {
    return(0);
  }
}

# init and open data base 
sub init_db {
  my($self) = shift;
  return if($self->{DBINIT}==1);
  
  my $db_exists;
  if(-f $self->{FNAME}) {$db_exists=1;}
  else                  {$db_exists=0;}

  # $self->LOGREPORT($self->{DBNAME},"open start",caller(),"");

  # connect to database
  print "$self->{INSTNAME}   LLmonDB_sqlite: connect to db $self->{FNAME} at ts=",time(),"\n"  if($self->{VERBOSE}==2);
  my $dbh = DBI->connect("dbi:SQLite:dbname=$self->{FNAME}","","",{PrintError => 1});
  $self->{DBH} = $dbh;
  $dbh->do("PRAGMA synchronous = OFF");
  $dbh->do("PRAGMA busy_timeout= 5000");
  $dbh->do("PRAGMA cache_size = 8000000");
  $dbh->do("PRAGMA journal_mode = WAL");
  $dbh->{sqlite_allow_multiple_statements}=1;
  # should be enabled only for write access !!! TODO
  #    $dbh->do("PRAGMA auto_vacuum = INCREMENTAL");
  $self->{DBINIT}=1;
  $self->{DBH}->{AutoCommit}=$self->{AUTOCOMMIT};
  
  my $eots=time();
  $self->{EOTS}=$eots;
  $self->mycommit();
  $self->LOGREPORT($self->{DBNAME},"open",caller(),"");
}


sub does_table_exists {
  my ($self,$table_name) = @_;

  my $sql = "SELECT name FROM sqlite_master WHERE (type='table') and (name='$table_name')";

  my $sth = $self->{DBH}->prepare($sql);
  $sth->execute();

  my @info = $sth->fetchrow_array;
  my $exists = scalar @info;

  return $exists;
}


# create a new table
sub create_table {
  my($self) = shift;
  my($table,$sqlcoldefs)=@_;
  my ($col);

  my $sql = "CREATE TABLE $table (";
  my @help;
  # print "TMPDEB: ",Dumper($sqlcoldefs);
  
  foreach $col (@{$sqlcoldefs->{collist}}) {
    push(@help, sprintf("%s %s",
                  $sqlcoldefs->{coldata}->{$col}->{name},
                  $sqlcoldefs->{coldata}->{$col}->{sql}));
  }
  $sql .= join(",",@help);
  $sql .= ")";
  
  $self->{DBH}->do($sql);
  $self->mycommit();

  print "\t   LLmonDB_sqlite: created table $table for $self->{FNAME} ($sql)\n";
  
  return();
}

# create a new table
sub recreate_table {
  my($self) = shift;
  my($table,$sqlcoldefs)=@_;
  my ($col,$newtable);
  $newtable=sprintf("_new_%s",$table);

  # 1. create new table
  my $sql = "CREATE TABLE $newtable (";
  my (@help,$rc);

  foreach $col (@{$sqlcoldefs->{collist}}) {
    push(@help, sprintf("%s %s",
                  $sqlcoldefs->{coldata}->{$col}->{name},
                  $sqlcoldefs->{coldata}->{$col}->{sql}));
  }
  $sql .= join(",",@help);
  $sql .= ")";
  $rc=$self->{DBH}->do($sql);
  if(!defined($rc)) {
    print STDERR "[recreate_table] \t   LLmonDB_sqlite: ERROR on last command, skip rest operation (created table $newtable, $sql)\n";
    return(-1);
  }
  print "\t   LLmonDB_sqlite: created table $newtable for ($sql) RC=$rc\n";

  # 2. copy data to new table
  $sql = "INSERT INTO $newtable SELECT ";
  $sql .= join(",",@{$sqlcoldefs->{collist}});
  $sql .= " from $table";
  $rc=$self->{DBH}->do($sql);
  if(!defined($rc)) {
    print STDERR "[recreate_table] \t   LLmonDB_sqlite: ERROR on last command, skip rest operation (copy data from $table to $newtable, $sql)\n";
    return(-1);
  }
  print "\t   LLmonDB_sqlite: copied data from $table to $newtable ($sql) RC=$rc\n";

  if(1) {
    # 3. drop old table
    $sql = "DROP TABLE $table";
    $rc=$self->{DBH}->do($sql);
    if(!defined($rc)) {
      print STDERR "[recreate_table] \t   LLmonDB_sqlite: ERROR on last command, skip rest operation (drop old table $table, $sql)\n";
      return(-1);
    }
    print "\t   LLmonDB_sqlite: drop old table $table ($sql) RC=$rc\n";

    # 4. rename new table
    $sql = "ALTER TABLE $newtable RENAME TO $table";
    $rc=$self->{DBH}->do($sql);
    if(!defined($rc)) {
      print STDERR "[recreate_table] \t   LLmonDB_sqlite: ERROR on last command, skip rest operation (rename $newtable to $table, $sql)\n";
      return(-1);
    }
    print "\t   LLmonDB_sqlite: drop old table ($sql) RC=$rc\n";
  }
  $self->mycommit();

  return();
}


# add column to a table
sub add_column {
  my($self) = shift;
  my($table,$col,$sqldef)=@_;

  my $sql = "ALTER TABLE $table ADD COLUMN $col $sqldef;";

  $self->{DBH}->do($sql);
  $self->mycommit();

  print "\t   LLmonDB_sqlite: add column $col to table $table for $self->{FNAME} ($sql)\n";

  return();
}


# remove column from a table
sub remove_column {
  my($self) = shift;
  my($table,$col)=@_;

  my $sql = "ALTER TABLE $table DROP COLUMN $col;";

  # $self->{DBH}->do($sql);
  # $self->mycommit();

  print "\t   LLmonDB_sqlite: delete column $col from table $table for $self->{FNAME} ($sql)\n";
  print "\t   LLmonDB_sqlite: WARNING: delete column $col from table $table for $self->{FNAME} NOT SUPPORTED in sqlite3\n";
  
  return();
}


# remove a new table
sub remove_table {
  my($self) = shift;
  my($table)=@_;

  my $sql = "DROP TABLE $table";
  $self->{DBH}->do($sql);
  $self->mycommit();

  print "\t   LLmonDB_sqlite: removed table $table for $self->{FNAME} ($sql)\n";
  
  return();
}


# create a new index table
sub create_index {
  my($self) = shift;
  my($table,$indextable,$colsref)=@_;

  my $sql = "CREATE INDEX $indextable on $table (";
  $sql .= join(",",@{$colsref});
  $sql .= ")";
  
  $self->{DBH}->do($sql);
  $self->mycommit();

  print "\t   LLmonDB_sqlite: created indextable $indextable for $self->{FNAME} ($sql)\n";
  
  return();
}


# create a new index table
sub remove_index {
  my($self) = shift;
  my($indextable)=@_;

  my $sql = "DROP INDEX $indextable";
  $self->{DBH}->do($sql);
  $self->mycommit();

  print "\t   LLmonDB_sqlite: remove indextable $indextable for $self->{FNAME} ($sql)\n";
  
  return();
}


sub close_db {
  my($self) = shift;
  
  return if($self->{DBINIT}==0);

  # disconnect from the database
  print "$self->{INSTNAME} disconnected from db $self->{FNAME} at ts=",time(),"\n" if($self->{VERBOSE}==2);
  $self->{DBH}->disconnect() if $self->{DBH};
  $self->{DBINIT}=0;
  
  $self->LOGREPORT($self->{DBNAME},"close",caller(),"");
}



# return pointer to array of table names
sub query_tables {
  my($self) = shift;

  my ($values_ref,@data);
  
  my $sql = "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY 1";

  my $sth = $self->{DBH}->prepare($sql);
  my $rc=$sth->execute();
  $self->LOGREPORT($self->{DBNAME},"E:query_tables",caller(),$DBI::errstr) if(!defined($rc));

  while(@data = $sth->fetchrow_array()) {
    push(@{$values_ref},$ data[0] );
  }
  return($values_ref);
}


# return pointer to array of columns names
sub query_columns {
  my($self) = shift;
  my($table) = @_;
  my ($retval,@data, $coldef);
  
  my $sql = "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name = '$table'";

  my $sth = $self->{DBH}->prepare($sql);
  my $rc=$sth->execute();
  $self->LOGREPORT($self->{DBNAME},"E:query_columns",caller(),$DBI::errstr) if(!defined($rc));

  if(@data = $sth->fetchrow_array()) {
    if($data[0]=~/^[^\(]+\((.*)\)$/) {
      my $cols=$1;
      foreach $coldef (split(/\s*,\s*/,$cols) ) {
        if($coldef=~/^\s*([^\s]+)\s(.+)\s*$/) {
          my ($name,$sql)=($1,$2);
          push(@{$retval->{collist}},$name);
          $retval->{coldata}->{$name}->{name}=$name;
          $retval->{coldata}->{$name}->{sql}=$sql;
        }
      }
    }
  }
  return($retval);
}


# return pointer to array of table names
sub query_index_tables {
  my($self) = shift;

  my ($values_ref,@data);
  
  my $sql = "SELECT name FROM sqlite_master WHERE type IN ('index') AND name NOT LIKE 'sqlite_%' ORDER BY 1";

  my $sth = $self->{DBH}->prepare($sql);
  my $rc=$sth->execute();
  $self->LOGREPORT($self->{DBNAME},"E:query_index_tables",caller(),$DBI::errstr) if(!defined($rc));

  while(@data = $sth->fetchrow_array()) {
    push(@{$values_ref},$ data[0] );
  }
  return($values_ref);
}


# return pointer to array of columns names
sub query_index_columns {
  my($self) = shift;
  my($table) = @_;
  my ($retval,@data);
  
  my $sql = "SELECT sql FROM sqlite_master WHERE type IN ('index') AND name = '$table'";

  my $sth = $self->{DBH}->prepare($sql);
  my $rc=$sth->execute();
  $self->LOGREPORT($self->{DBNAME},"E:query_index_columns",caller(),$DBI::errstr) if(!defined($rc));

  if(@data = $sth->fetchrow_array()) {
    if($data[0]=~/\(([^\)]*)\)/) {
      my $cols=$1;
      foreach my $col (split(/\s*,\s*/,$cols) ) {
        push(@{$retval->{collist}},$col);
      }
    }
  }
  return($retval);
}


# start insert sequence
sub start_insert_sequence {
  my($self) = shift;
  my($table,$colsref) = @_;
  my @placeholders=map { '?' } 1.. scalar @{$colsref};
  my $sql = sprintf('INSERT INTO %s (%s) VALUES (%s)',
                    $table,
                    join(",",@{$colsref}),
                    join(",",@placeholders),
                    );
  my $sth = $self->{DBH}->prepare($sql);

  $self->{DBH}->begin_work();
  printf("\t   LLmonDB_sqlite: start_insert_sequence %s (sql: %s...)\n",$table,substr($sql,0,16)) if($debug>=3);
  return($sth);
}

# end insert sequence
sub insert_sequence {
  my($self) = shift;
  my($sth,$dataref) = @_;
  my $rc=$sth->execute( @{$dataref} );
  $self->LOGREPORT($self->{DBNAME},"E:insert_sequence",caller(),$DBI::errstr) if(!defined($rc));
  return($rc);

  $self->{A}=0;  #dummy statement to avoid unused warning
}


# end insert sequence
sub end_insert_sequence {
  my($self) = shift;
  my($sth) = @_;

  $sth->finish();
  $self->{DBH}->commit();
  # $self->mycommit();
  print "\t   LLmonDB_sqlite: end_insert_sequence\n" if($debug>=3);
  return();
}


# remove table contents
sub remove_contents {
  my($self) = shift;
  my($table) = @_;

  my $sql="delete from $table";
  my $rc=$self->{DBH}->do($sql);
  printf("\t   LLmonDB_sqlite: removed (sql: %s...)\n",substr($sql,0,16)) if($debug>=3);
  $self->mycommit();
  
  return($rc);
}


# execute SQL-commands
sub execute_sql {
  my($self) = shift;
  my($sqllist) = @_;
  my $rc_all=0;
  foreach my $sql (split(/\s*;\s*/,$sqllist)) {
    # printf("\t   LLmonDB_sqlite: executed (sql: %s)\n",$sql);
    my $rc=$self->{DBH}->do($sql);
    $self->LOGREPORT($self->{DBNAME},"E:execute_sql",caller(),$DBI::errstr) if(!defined($rc));
    printf(STDERR "[execute_sql] \t   LLmonDB_sqlite: executed sql: %s \n",$sql) if(!defined($rc));
    $rc_all+=$rc  if(defined($rc));
  }
  $self->mycommit();
  
  return($rc_all);
}


# return pointer to result of query
sub query {
  my($self) = shift;
  my($table,$optsref) = @_;
  my ($retval);

  # print Dumper($optsref);
  my $type="aref_of_aref";
  $type=$optsref->{type} if(exists($optsref->{type}));

  $type="hash_values" if ($type eq "hash_single_value"); # old name
  if($type eq "hash_values") {
    $retval=$self->query_hash_values_table($table,
                              $optsref->{hash_keys},
                              $optsref->{hash_value},
                              $optsref->{where},
                              $optsref->{sql}
                            );
  } elsif($type eq "get_arrayref_of_arrayref") {
    $retval=$self->query_arrayref_of_arrayref_table($table,
                                        $optsref->{where},
                                        $optsref->{sql}
                                      );
  } elsif($type eq "get_arrayref_of_hashref") {
    $retval=$self->query_arrayref_of_hashref_table($table,
                                      $optsref->{where},
                                      $optsref->{sql}
                                    );
  } elsif($type eq "get_min") {
    $retval=$self->query_get_min($table,
                        $optsref->{hash_key},
                        $optsref->{where},
                      );
  } elsif($type eq "get_max") {
    $retval=$self->query_get_max($table,
                        $optsref->{hash_key},
                        $optsref->{where},
                      );
  } elsif($type eq "get_count") {
    $retval=$self->query_get_count($table,
                        $optsref->{where},
                        );
  } elsif($type eq "get_execute") {
    $retval=$self->query_get_execute($table,
                          $optsref->{columns},
                          $optsref->{where},
                          $optsref->{sql},
                          $optsref->{attach},
                          $optsref->{execute},
                          "array"
                        );

  } elsif($type eq "get_execute_hash") {
    $retval=$self->query_get_execute($table,
                        $optsref->{columns},
                        $optsref->{where},
                        $optsref->{sql},
                        $optsref->{attach},
                        $optsref->{execute},
                        "hash"
                      );

  } else {
    print STDERR "[query] \t   LLmonDB_sqlite: ERROR in query, unknown type $type\n";
  }
  return($retval);
}

sub query_hash_values_table {
  my($self) = shift;
  my($table,$nkeys,$nvalue,$where,$qsql) = @_;
  my ($retval,$keylist);

  if(ref(\$nkeys) eq "SCALAR") {
    $keylist=$nkeys;
  } else {
    $keylist=join(",",@{$nkeys});
  } 
  # print "query_hash_values_table: $nkeys (".ref(\$nkeys).") -> $keylist\n";
  my $sql;
  if(!defined($qsql)) {
    if(!defined($table)) {
      printf(STDERR "[query_hash_values_table] \t   LLmonDB_sqlite: ERROR cannot prepare query_hash_values_table, [table not defined] ($nkeys,$nvalue,$where,$qsql,%s,%s)\n",caller());
      return(undef);
    }
    $sql = "SELECT $keylist,$nvalue FROM $table";
    $sql.=" WHERE $where" if($where);
  } else {
    $sql=$qsql;
    # print "query_hash_values_table: sql=$sql\n";
  }
  # print "query_hash_values_table: $sql\n";
  my $sth = $self->{DBH}->prepare($sql);
  if(!$sth) {
    printf(STDERR "[query_hash_values_table] \t   LLmonDB_sqlite: ERROR cannot prepare query_hash_values_table(table: $table sql: %s...)\n",substr($sql,0,240));
    return(undef);
  }
  $sth->execute();

  # $self->{DBH}->{FetchHashKeyName} = 'NAME_lc'; # use lowercase names
  $retval = $sth->fetchall_hashref($nkeys);

  print "\t   LLmonDB_sqlite: query_hash_single_table $sql\n" if($debug>=3);
  # print Dumper($retval);
  return($retval);
}


sub query_arrayref_of_arrayref_table {
  my($self) = shift;
  my($table,$where,$qsql) = @_;
  my ($retval);

  my $sql;
  if(!defined($qsql)) {
    $sql = "SELECT * FROM $table";
    $sql.=" WHERE $where" if($where);
  } else {
    $sql=$qsql;
    # print "query_hash_values_table: sql=$sql\n";
  }
  print "query_arrays_table: $sql\n";
  my $sth = $self->{DBH}->prepare($sql);
  if(!$sth) {
    printf(STDERR "[query_arrayref_of_arrayref_table] \t   LLmonDB_sqlite: ERROR cannot prepare query_arrayref_of_arrayref_table(table: $table sql: %s...)\n",substr($sql,0,120));
    return(undef);
  }
  $sth->execute();

  # $self->{DBH}->{FetchHashKeyName} = 'NAME_lc'; # use lowercase names
  $retval = $sth->fetchall_arrayref();

  print "\t   LLmonDB_sqlite: query_arrays_table $sql\n" if($debug>=3);
  # print Dumper($retval);
  return($retval);
}


sub query_arrayref_of_hashref_table {
  my($self) = shift;
  my($table,$where,$qsql) = @_;
  my ($retval);

  my $sql;
  if(!defined($qsql)) {
    $sql = "SELECT * FROM $table";
    $sql.=" WHERE $where" if($where);
  } else {
    $sql=$qsql;
    # print "query_hash_values_table: sql=$sql\n";
  }
  # print "query_arrays_table: $sql\n";
  my $sth = $self->{DBH}->prepare($sql);
  if(!$sth) {
    printf(STDERR "[query_arrayref_of_hashref_table] \t   LLmonDB_sqlite: ERROR cannot prepare query_arrayref_of_hashref_table(table: $table sql: %s...)\n",substr($sql,0,120));
    return(undef);
  }
  $sth->execute();

  # $self->{DBH}->{FetchHashKeyName} = 'NAME_lc'; # use lowercase names
  $retval = $sth->fetchall_arrayref({});

  print "\t   LLmonDB_sqlite: query_arrays_table $sql\n" if($debug>=3);
  # print Dumper($retval);
  return($retval);
}


sub query_get_min {
  my($self) = shift;
  my($table,$key,$where) = @_;
  my ($retval);
    
  my $sql = "SELECT min($key) FROM $table";
  $sql.=" WHERE $where" if($where);
  
  my $sth = $self->{DBH}->prepare($sql);
  if(!$sth) {
    printf(STDERR "[query_get_min] \t   LLmonDB_sqlite: ERROR cannot prepare query_get_min(table: $table sql: %s...)\n",substr($sql,0,120));
    return(undef);
  }
  $sth->execute();
  if (my @data = $sth->fetchrow_array()) {
    $retval=$data[0];
  } else {
    $retval=-1;
  }
  print "\t   LLmonDB_sqlite: query_get_min $sql\n" if($debug>=3);
  # print Dumper($retval);
  return($retval);
}


sub query_get_max {
  my($self) = shift;
  my($table,$key,$where) = @_;
  my ($retval);
    
  my $sql = "SELECT max($key) FROM $table";
  $sql.=" WHERE $where" if($where);
  
  my $sth = $self->{DBH}->prepare($sql);
  if(!$sth) {
    printf(STDERR "[query_get_max] \t   LLmonDB_sqlite: ERROR cannot prepare query_get_max(table: $table sql: %s...)\n",substr($sql,0,120));
    return(undef);
  }
  $sth->execute();
  if (my @data = $sth->fetchrow_array()) {
    $retval=$data[0];
  } else {
    $retval=-1;
  }
  print "\t   LLmonDB_sqlite: query_get_max $sql\n" if($debug>=3);
  # print Dumper($retval);
  return($retval);
}


sub query_get_count {
  my($self) = shift;
  my($table,$where) = @_;
  my ($retval);
    
  my $sql = "SELECT count(*) FROM $table";
  $sql.=" WHERE $where" if($where);
  
  my $sth = $self->{DBH}->prepare($sql);
  if(!$sth) {
    printf(STDERR "[query_get_count] \t   LLmonDB_sqlite: ERROR cannot prepare query_get_count(table: $table sql: %s...)\n",substr($sql,0,120));
    return(undef);
  }
  my $rc=$sth->execute();
  $self->LOGREPORT($self->{DBNAME},"E:query_get_count",caller(),$DBI::errstr) if(!defined($rc));
  if (my @data = $sth->fetchrow_array()) {
    $retval=$data[0];
  } else {
    $retval=-1;
  }
  print "\t   LLmonDB_sqlite: query_get_count $sql\n" if($debug>=3);
  # print Dumper($retval);
  return($retval);
}


sub query_get_execute {
  my($self) = shift;
  my($table,$columns,$where,$qsql,$attach,$funcref,$qtype) = @_;
  my ($retval,@data,$hashref,$sql);

  if(!defined($qsql)) {
    $sql = sprintf ("SELECT %s FROM %s",join(",",@{$columns}),$table);
    $sql.=" WHERE $where" if($where);
  } else {
    $sql=$qsql;
  }

  if(defined($attach)) {
    # END TRANSACTION;
    
    my $endtransaction="";
    $endtransaction="END TRANSACTION;" if(!$self->{AUTOCOMMIT});
    
    my $attsql=sprintf("${endtransaction}ATTACH DATABASE '%s/LLmonDB_%s.sqlite' as %s",
                        $self->{DBDIR}, $attach, $attach);
    my $arc=$self->{DBH}->do($attsql);
    print "\t   LLmonDB_sqlite: query_get_execute do $attsql rc=$arc\n" if($debug>=3);
    $self->LOGREPORT($self->{DBNAME},"E:query_get_execute: attach to $attach",caller(),$DBI::errstr) if(!defined($arc));
    $self->LOGREPORT($attach,"E:query_get_execute: attach from $self->{DBNAME}",caller(),$DBI::errstr) if(!defined($arc));
    # $self->get_database_structure();
  }
  
  my $sth = $self->{DBH}->prepare($sql);
  if(!$sth) {
    printf(STDERR "[query_get_execute] \t   LLmonDB_sqlite: ERROR cannot prepare query_get_execute(table: $table sql: %s...)\n",substr($sql,0,120));
    return(undef);
  }
  my $rc=$sth->execute();
  print "\t   LLmonDB_sqlite: query_get_execute execute rc=$rc\n" if($debug>=3);
  $self->LOGREPORT($self->{DBNAME},"E:query_get_execute",caller(),$DBI::errstr) if(!defined($rc));
  my $c=0;
  if($qtype eq "array") {
    while (@data = $sth->fetchrow_array()) {
      &$funcref(@data); $c++;
    }
  } elsif($qtype eq "hash") { 
    while ($hashref = $sth->fetchrow_hashref()) {
      &$funcref($hashref); $c++;
    }
  } else {
    print STDERR "[query_get_execute] \t   LLmonDB_sqlite: ERROR in query_get_execute, unknown qtype $qtype\n";
  }
  $retval=$c;

  if(defined($attach)) {
    my $endtransaction="";
    $endtransaction="END TRANSACTION;" if(!$self->{AUTOCOMMIT});
    my $attsql=sprintf("${endtransaction}DETACH DATABASE %s", $attach);
    my $drc=$self->{DBH}->do($attsql);
    $self->LOGREPORT($self->{DBNAME},"E:query_get_execute: dettach to $attach",caller(),$DBI::errstr) if(!defined($drc));
    $self->LOGREPORT($attach,"E:query_get_execute: dettach from $self->{DBNAME}",caller(),$DBI::errstr) if(!defined($drc));
  }

  print "\t   LLmonDB_sqlite: query_get_execute $sql ($c entries)\n" if($debug>=3);
  return($retval);
}


sub get_database_structure {
  my($self) = shift;

  # Return the structure of the table execution_host
  my $sth = $self->{DBH}->prepare('pragma database_list');
  my $rc=$sth->execute();
  $self->LOGREPORT($self->{DBNAME},"E:get_database_structure",caller(),$DBI::errstr) if(!defined($rc));
  my @struct;
  while (my $row = $sth->fetchrow_arrayref()) {
    push @struct, @$row[1];
  }
  print "get_database_structure:",Dumper(\@struct);

  return @struct;
}

# return pointer to result of query
sub delete {
  my($self) = shift;
  my($table,$optsref) = @_;
  my ($retval);

  my $type="all_rows";
  $type=$optsref->{type} if(exists($optsref->{type}));

  if($type eq "all_rows") {
    $retval=$self->delete_all_rows($table);
  } elsif($type eq "some_rows") {
    $retval=$self->delete_some_rows($table,
                          $optsref->{where}
                        );
  } else {
    print STDERR "[delete] \t   LLmonDB_sqlite: ERROR in delete, unknown type $type\n";
  }
  $self->LOGREPORT($self->{DBNAME},"E:delete",caller(),$DBI::errstr) if(!defined($retval));
  return($retval);
}

sub delete_all_rows {
  my($self) = shift;
  my($table) = @_;
  my ($retval);
  
  my $sql="DELETE FROM $table";
  
  my $rc=$self->{DBH}->do($sql);
  $self->LOGREPORT($self->{DBNAME},"E:delete_all_rows",caller(),$DBI::errstr) if(!defined($rc));
  $self->mycommit();
  $retval=$rc;
  print "\t   LLmonDB_sqlite: delete_all_entries $sql ($rc entries)\n" if($debug>=3);
  return($retval);
}

sub delete_some_rows {
  my($self) = shift;
  my($table,$where) = @_;
  my ($retval);
  
  my $sql="DELETE FROM $table";
  $sql.=" WHERE $where" if($where);
  my $rc=$self->{DBH}->do($sql);
  $self->LOGREPORT($self->{DBNAME},"E:delete_some_rows",caller(),$DBI::errstr) if(!defined($rc));
  if(!defined($rc)) {
    print STDERR "[delete_some_rows] \t   LLmonDB_sqlite: ERROR in delete_some_rows, sql=$sql\n";
  }
  $self->mycommit();
  $retval=$rc;
  print "\t   LLmonDB_sqlite: delete_some_entries $sql ($rc entries)\n" if($debug>=3);
  return($retval);
}

1;
