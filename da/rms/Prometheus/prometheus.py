#!/usr/bin/env python3
# Copyright (c) 2023 Forschungszentrum Juelich GmbH.
# This file is part of LLview. 
#
# This is an open source software distributed under the GPLv3 license. More information see the LICENSE file at the top level.
#
# Contributions must follow the Contributor License Agreement. More information see the CONTRIBUTING.md file at the top level.
#
# Contributors:
#    Filipe Guimarães (Forschungszentrum Juelich GmbH)

import argparse
import logging
import time
from datetime import datetime
import re
import os
import sys
import math
import csv
import ast
import getpass
import requests
from copy import deepcopy
from subprocess import check_output


def cores(options: dict, cores_info) -> dict:
  """
  Specific function to add extra items to gpus entries
  """
  log = logging.getLogger('logger')

  # Getting information from the steps
  log.info("Adding extra information for cores...\n")

  coressextra = {}
  # Updating the jobs dictionary by adding or removing keys
  for node,coresinfo in cores_info.items():
    coressextra.setdefault(node,{})
    # Adding usage per core
    coressextra[node]['percore'] = ','.join([f"{core}:{1-coreidle:g}" for core,coreidle in sorted(coresinfo['coreidle'].items())])

    del coresinfo['coreidle']

  return coressextra

def cpus(options: dict, cpus_info) -> dict:
  """
  Specific function to add extra items to gpus entries
  """
  log = logging.getLogger('logger')

  # Getting information from the steps
  log.info("Adding extra information for cpus...\n")

  cpusextra = {}
  # Updating the jobs dictionary by adding or removing keys
  for node,cpuinfo in cpus_info.items():
    # print(node,coreinfo)
    cpusextra.setdefault(node,{})
    cpusextra[node]['usage'] = 0
    cpusextra[node]['physcoresused'] = 0
    cpusextra[node]['logiccoresused'] = 0
    ncores = len(cpuinfo['coreidle'].keys())/2
    for core,coreidle in cpuinfo['coreidle'].items():
      cpusextra[node]['usage'] += 1 - coreidle
      if core < ncores:
        cpusextra[node]['physcoresused']+= int(coreidle<0.75)
      else:
        cpusextra[node]['logiccoresused']+= int(coreidle<0.75)

    cpusextra[node]['usage'] = cpusextra[node]['usage']/ncores

    del cpuinfo['coreidle']
  return cpusextra


def gpu(options: dict, gpus_info) -> dict:
  """
  Specific function to add extra items to gpus entries
  """
  log = logging.getLogger('logger')

  # Getting information from the steps
  log.info("Adding extra information for gpus...\n")

  log.debug(f"Performing query to check if node is up: 'sum by(instance) (up{{job=~\"node-compute.*\"}})' \n")
  query = 'sum by(instance) (up{job=~"node-compute.*"})'
  url = f"https://{gpus_info._host}/api/v1/query?query={requests.utils.quote(query)}"
  log.debug(f"{url}\n")

  # Querying Prometheus server
  if gpus_info._token:
    # with token
    headers = {'accept': 'application/json', 'Authorization': gpus_info._token}
    r = requests.get(url, headers=headers, timeout=(5,10))
  else:
    # with credentials
    credentials = None
    if gpus_info._user and gpus_info._pass:
      credentials = (gpus_info._user, gpus_info._pass)
    r = requests.get(url, auth=credentials, timeout=(5,10))

  # If current query does not succeed, log error and skip next steps
  if not r.ok:
    log.error(f"Status <{r.status_code}> with query {url}")
    return

  # Parsing imporant part of the query:
  data = {}
  for entry in r.json()['data']['result']:
    data[entry['metric']['instance']] = int(entry['value'][1])

  gpusextra = {}
  # Updating the gpus dictionary by adding or removing keys
  for gpuname,gpuinfo in gpus_info.items():
    gpusextra.setdefault(gpuname,{})
    gpusextra[gpuname]['state'] = 'Running' if gpuinfo['id'] in data and data[gpuinfo['id']] else 'Down'
    gpusextra[gpuname]['features'] = f"GPU{gpuinfo['id'][-1:]}"
  return gpusextra

class Info:
  """
  Class that stores and processes information from Slurm output  
  """
  def __init__(self,hostname="",username=None,password=None,token=None):
    self._raw = {}  # Dictionary with parsed raw information
    self._dict = {} # Dictionary with modified information (which is output to LML)
    self._host = os.path.expandvars(hostname)
    self._user = username
    self._pass = password
    self._token = token
    self.log   = logging.getLogger('logger')

  def __add__(self, other):
    first = self
    second = other
    first._raw |= second._raw
    first.add(second._dict)
    return first

  def __iter__(self):
    return (t for t in self._dict.keys())
    
  def __len__(self):
    return len(self._dict)

  def items(self):
    return self._dict.items()

  def __delitem__(self,key):
    del self._dict[key]

  def add(self, to_add: dict, add_to=None):
    """
    (Deep) Merge dictionary 'to_add' into internal 'self._dict'
    """
    # self._dict |= dict
    if not add_to:
      add_to = self._dict
    for bk, bv in to_add.items():
      av = add_to.get(bk)
      if isinstance(av, dict) and isinstance(bv, dict):
        self.add(bv,add_to=av)
      else:
        add_to[bk] = deepcopy(bv)
    return

  def empty(self):
    """
    Check if internal dict is empty: Boolean function that returns True if _dict is empty
    """
    return not bool(self._dict)

  def query(self, metrics, prefix="", stype="", cached_queries = {}):
    """
    This function will loop through the metrics given in the list 'metrics'
    and perform them in the server defined in self.
    The values are put in the dict self._raw
    """

    # Looping over all metrics that should be put into the file
    for name,metric in metrics.items():
      # Recovering data from cached query when that is present
      if name in cached_queries:
        self.log.info(f"Query {name} is cached, recovering data without querying again...\n")
        data = cached_queries[name]
      else:
        if 'query' not in metric:
          self.log.debug(f"No query to be done for {name}\n")
          continue

        if not self._host:
          self.log.error(f"No hostname defined for current server. Query {name} cannot be done! Skipping...")
          continue

        self.log.debug(f"Performing query: {metric['query']}\n")
        url = f"https://{self._host}/api/v1/query?query={requests.utils.quote(metric['query'])}"
        self.log.debug(f"{url}\n")

        # Querying Prometheus server
        if self._token:
          # with token
          headers = {'accept': 'application/json', 'Authorization': self._token}
          r = requests.get(url, headers=headers, timeout=(5,10))
        else:
          # with credentials
          credentials = None
          if self._user and self._pass:
            credentials = (self._user, self._pass)
          r = requests.get(url, auth=credentials, timeout=(5,10))

        # If current query does not succeed, log error and continue to next query
        if not r.ok:
          self.log.error(f"Status <{r.status_code}> with query {url}")
          continue

        # Getting data from returned result of query
        data = r.json()['data']['result']

        if ('cache' in metric) and metric['cache']:
          cached_queries[name] = data

      # Looping over all instances (nodes/gpus) in current queried result
      for instance in data:
        # Getting name of the node
        pid = instance['metric']['instance']
        gpu = None
        if 'device' in instance['metric']:
          # If 'device' key is present, this is a gpu
          gpu = int(instance['metric']['device'].replace('nvidia',''))
        id = f"{pid}" + ( f"_{gpu:02d}" if isinstance(gpu,int) else "" )
        self._raw.setdefault(id,{})
        # Getting value
        if 'cpu' in instance['metric']:
          self._raw[id].setdefault(name,{})
          self._raw[id][name][ast.literal_eval(instance['metric']['cpu'])] = ast.literal_eval(instance['value'][1])
          if ('min' in metric) and self._raw[id][name][ast.literal_eval(instance['metric']['cpu'])] < metric['min']: self._raw[id][name][ast.literal_eval(instance['metric']['cpu'])] = metric['min']
          if ('max' in metric) and self._raw[id][name][ast.literal_eval(instance['metric']['cpu'])] > metric['max']: self._raw[id][name][ast.literal_eval(instance['metric']['cpu'])] = metric['max']
        else:
          self._raw[id][name] = ast.literal_eval(instance['value'][1])
          if ('min' in metric) and self._raw[id][name] < metric['min']: self._raw[id][name] = metric['min']
          if ('max' in metric) and self._raw[id][name] > metric['max']: self._raw[id][name] = metric['max']
        if 'factor' in metric:
          self._raw[id][name] *= metric['factor']
        # Adding extra keys 
        self._raw[id][f'{prefix}_ts' if prefix else 'ts'] = instance['value'][0]
        if gpu:
          self._raw[id]['pid'] = pid
          self._raw[id]['id'] = id
          self._raw[id]['feature'] = f"GPU{gpu}"
        else:
          self._raw[id]['id'] = pid

    for instance in self._raw:
      # Adding prefix and type of the unit, when given in the input
      if prefix:
          self._raw[instance]["__prefix"] = prefix
      if stype:
        self._raw[instance]["__type"] = stype
      # Adding default values for missing metrics
      for name,metric in metrics.items():
        if name not in self._raw[instance] and 'default' in metric:
          self._raw[instance][name] = metric['default']

    self._dict |= self._raw
    return cached_queries


  def parse(self, cmd, timestamp="", prefix="", stype=""):
    """
    This function parses the output of Slurm commands
    and returns them in a dictionary
    """

    # Getting Slurm raw output
    rawoutput = check_output(cmd, shell=True, text=True)
    # 'scontrol' has an output that is different from
    # 'sacct' and 'sacctmgr' (the latter are csv-like)
    if("scontrol" in cmd):
      # If result is empty, return
      if (re.match("No (.*) in the system",rawoutput)):
        self.log.warning(rawoutput.split("\n")[0]+"\n")
        return
      # Getting unit to be parsed from first keyword
      unitname = re.match("(\w+)",rawoutput).group(1)
      self.log.debug(f"Parsing units of {unitname}...\n")
      units = re.findall(f"({unitname}[\s\S]+?)\n\n",rawoutput)
      for unit in units:
        self.parse_unit_block(unit, unitname, prefix, stype)
    else:
      units = list(csv.DictReader(rawoutput.splitlines(), delimiter='|'))
      if len(units) == 0:
        self.log.warning(f"No output units from command {cmd}\n")
        return
      # Getting unit to be parsed from first keyword
      unitname = re.match("(\w+)",rawoutput).group(1)
      self.log.debug(f"Parsing units of {unitname}...\n")
      for unit in units:
        current_unit = unit[unitname]
        self._raw[current_unit] = {}
        # Adding prefix and type of the unit, when given in the input
        if prefix:
          self._raw[current_unit]["__prefix"] = prefix
        if stype:
          self._raw[current_unit]["__type"] = stype
        for key,value in unit.items():
          self.add_value(key,value,self._raw[current_unit])

    self._dict |= self._raw
    return

  def add_value(self,key,value,dict):
    """
    Function to add (key,value) pair to dict. It is separate to be easier to adapt
    (e.g., to not include empty keys)
    """
    dict[key] = value if value != "(null)" else ""
    return

  def parse_unit_block(self, unit, unitname, prefix, stype):
    """
    Parse each of the blocks returned by Slurm into the internal dictionary self._raw
    """
    # self.log.debug(f"Unit: \n{unit}\n")
    lines = unit.split("\n")
    # first line treated differently to get the 'unit' name and avoid unnecessary comparisons
    for pair in lines[0].strip().split(' '):
      key, value = pair.split('=',1)
      if key == unitname:
        current_unit = value
        self._raw[current_unit] = {}
        # Adding prefix and type of the unit, when given in the input
        if stype:
          self._raw[current_unit]["__prefix"] = prefix
        if stype:
          self._raw[current_unit]["__type"] = stype
      # JobName must be treated separately, as it does not occupy the full line
      # and it may contain '=' and ' '
      elif key == "JobName":
        value = re.search(".*JobName=(.*)$",lines[0].strip()).group(1)
        self._raw[current_unit][key] = value
        break
      self.add_value(key,value,self._raw[current_unit])

    # Other lines must be checked if there are more than one item per line
    # When one item per line, it must be considered that it may include '=' in 'value'
    for line in [_.strip() for _ in lines[1:]]:
      # Skip empty lines
      if not line: continue
      self.log.debug(f"Parsing line: {line}\n")
      # It is necessary to handle lines that can contain '=' and ' ' in 'value' first
      if len(splitted := line.split('=',1)) == 2: # Checking if line is splittable on "=" sign
        key,value = splitted
      else:  # If not, split on ":"
        key,value = line.split(":",1)
      # Here must be all fields that can contain '=' and ' ', otherwise it may break the workflow below 
      if key in ['Comment','Reason','Command','WorkDir','StdErr','StdIn','StdOut','TRES','OS']: 
        self.add_value(key,value,self._raw[current_unit])
        continue
      # Now the pairs are separated by space
      for pair in line.split(' '):
        if len(splitted := pair.split('=',1)) == 2: # Checking if line is splittable on "=" sign
          key,value = splitted
        else:  # If not, split on ":"
          key,value = pair.split(":",1)
        if key in ['Dist']: #'JobName'
          self._raw[current_unit][key] = line.split(f'{key}=',1)[1]
          break
        self.add_value(key,value,self._raw[current_unit])
    return

  def apply_pattern(self,exclude="",include=""):
    """
    Loops over all units in self._dict to:
    - remove items that match 'exclude'
    - keep only items that match 'include'
    """
    to_remove = []
    for unitname,unit in self._dict.items():
      if exclude and self.check_unit(unitname,unit,exclude,text="excluded") == True:
        to_remove.append(unitname)
      if include and self.check_unit(unitname,unit,include,text="included") == False:
        to_remove.append(unitname)
    for unitname in to_remove:
      del self._dict[unitname]
    return
  
  def check_unit(self,unitname,unit,pattern,text="included/excluded"):
    """
    Check 'current_unit' name with rules for exclusion or inclusion. (exclusion is applied first)
    Returns True if unit is to be skipped
    """
    if isinstance(pattern,str): # If rule is a simple string
      if re.match(pattern, unitname):
        self.log.debug(f"Unit {unitname} is {text} due to {pattern} rule\n")
        return True
    elif isinstance(pattern,list): # If list of rules
      for pat in pattern: # loop over list - that can be strings or dictionaries
        if isinstance(pat,str): # If item in list is a simple string
          if re.match(pat, unitname):
            self.log.debug(f"Unit {unitname} is {text} due to {pat} rule in list\n")
            return True
        elif isinstance(pat,dict): # If item in list is a dictionary
          for key,value in pat.items():
            if isinstance(value,str): # if dictionary value is a simple string
              if (key in unit) and re.match(value, unit[key]):
                self.log.debug(f"Unit {unitname} is {text} due to {value} rule in {key} key of list\n")
                return True
            elif isinstance(value,list): # if dictionary value is a list
              for v in value:
                if (key in unit) and re.match(v, unit[key]): # At this point, v in list can only be a string
                  self.log.debug(f"Unit {unitname} is {text} due to {v} rule in list of {key} key of list\n")
                  return True
    elif isinstance(pattern,dict): # If dictionary with rules
      for key,value in pattern.items():
        if isinstance(value,str): # if dictionary value is a simple string
          if (key in unit) and re.match(value, unit[key]):
            self.log.debug(f"Unit {unitname} is {text} due to {value} rule in {key} key\n")
            return True
        elif isinstance(value,list): # if dictionary value is a list
          for v in value:
            if (key in unit) and re.match(v, unit[key]): # At this point, v in list can only be a string
              self.log.debug(f"Unit {unitname} is {text} due to {v} rule in list of {key} key\n")
              return True            
    return False

  def map(self, mapping_dict):
    """
    Map the dictionary using (key,value) pair in mapping_dict
    (Keys that are not present are removed)
    """
    new_dict = {}
    skip_keys = set()
    for unit,item in self._dict.items():
      new_dict[unit] = {}
      for key,map in mapping_dict.items():
        # Checking if key to be modified is in object
        if key not in item:
          skip_keys.add(key)
          continue
        new_dict[unit][map] = item[key]
      # Copying also internal keys that are used in the LML
      if '__type' in item:
        new_dict[unit]['__type'] = item['__type']
      if '__id' in item:
        new_dict[unit]['__id'] = item['__id']
      if '__prefix' in item:
        new_dict[unit]['__prefix'] = item['__prefix']
    if skip_keys:
      self.log.warning(f"Skipped mapping keys (at least on one node): {', '.join(skip_keys)}\n")
    self._dict = new_dict
    return

  def modify(self, modify_dict):
    """
    Modify the dictionary using functions given in modify_dict
    """
    skipped_keys = set()
    for item in self._dict.values():
      for key,modify in modify_dict.items():
        # Checking if key to be modified is in object
        if key not in item:
          skipped_keys.add(key)
          continue
        if isinstance(modify,str):
          for funcname in [_.strip() for _ in modify.split(',')]:
            try:
              func = globals()[funcname]
              item[key] = func(item[key])
            except KeyError:
              self.log.error(f"Function {funcname} is not defined. Skipping it and keeping value {item[key]}\n")
        elif isinstance(modify,list):
          for funcname in modify:
            try:
              func = globals()[funcname]
              item[key] = func(item[key])
            except KeyError:
              self.log.error(f"Function {funcname} is not defined. Skipping it and keeping value {item[key]}\n")
    if skipped_keys:
      self.log.warning(f"Skipped modifying keys (at least on one node): {', '.join(skipped_keys)}\n")
    return

  def to_LML(self, filename, prefix="", stype=""):
    """
    Create LML output file 'filename' using
    information of self._dict
    """
    self.log.info(f"Writing LML data to {filename}... ")
    # Creating folder if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Opening LML file
    with open(filename,"w") as file:
      # Writing initial XML preamble
      file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" )
      file.write("<lml:lgui xmlns:lml=\"http://eclipse.org/ptp/lml\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n" )
      file.write("    xsi:schemaLocation=\"http://eclipse.org/ptp/lml http://eclipse.org/ptp/schemas/v1.1/lgui.xsd\"\n" )
      file.write("    version=\"1.1\">\n" )

      # Creating first list of objects
      file.write("<objects>\n" )
      digits = int(math.log10(len(self._dict)))+1 if len(self._dict)>0 else 1
      i = 700 if "percore" in filename else 0
      for key,item in self._dict.items():
        if "__id" not in item:
          item["__id"] = f'{prefix if prefix else item["__prefix"]}{i:0{digits}d}'
          i += 1
        file.write(f'<object id=\"{item["__id"]}\" name=\"{key}\" type=\"{stype if stype else item["__type"]}\"/>\n')
      file.write("</objects>\n")

      # Writing detailed information for each object
      file.write("<information>\n")
      # Counter of the number of items that define each object
      i = 0
      # Looping over the items
      for item in self._dict.values():
        # The objects are unique for the combination {jobid,path}
        file.write(f'<info oid=\"{item["__id"]}\" type=\"short\">\n')
        # Looping over the quantities obtained in this item
        for key,value in item.items():
          # The __nelems_{type} is used to indicate to DBupdate the number of elements - important when the file is empty
          if key.startswith('__nelems'): 
            file.write(" <data key={:24s} value=\"{}\"/>\n".format('\"'+str(key)+'\"',value))
            continue
          if key.startswith('__'): continue
          if (not isinstance(value,str)) or (value != ""):
          # if (value) and (value != "0"):
            # Replacing double quotes with single quotes to avoid problems importing the values
            file.write(" <data key={:24s} value=\"{}\"/>\n".format('\"'+str(key)+'\"',value.replace('"', "'") if isinstance(value, str) else value))
        # if ts:
        #   file.write(" <data key={:24s} value=\"{}\"/>\n".format('\"ts\"',ts))

        file.write(f"</info>\n")
        i += 1

      file.write("</information>\n" )
      file.write("</lml:lgui>\n" )

    log_continue(self.log,"Finished!")

    return


def log_continue(log,message):
  """
  Change formatter to write a continuation 'message' on the logger 'log' and then change the format back
  """
  for handler in log.handlers:
    handler.setFormatter(CustomFormatter("%(message)s (%(lineno)-3d)[%(asctime)s]\n",datefmt=log_config['datefmt']))

  log.info(message)

  for handler in log.handlers:
    handler.setFormatter(CustomFormatter(log_config['format'],datefmt=log_config['datefmt']))
  return


def get_token(username,password,config):
  """
  Get token to be used in requests
  """
  log = logging.getLogger('logger')
  # Build Auth URI for token
  token_endpoint = config['endpoint']

  # Preparing data to request token
  data = {
      'username': username,
      'password': password,
      'grant_type': 'password',
      'client_id': config['client_id']
  }
  headers = {'content-type': 'application/x-www-form-urlencoded'}

  # Get token endpoint
  log.info(f"Requesting token from POST method via {token_endpoint}\n")
  token_request = requests.post(token_endpoint, data=data, headers=headers)

  if not token_request.ok:
    log.error(f'Token request not successful (return code {token_request.status_code})! Creating empty LML...\n')
    return ""
  log.debug(f"token_request: {token_request.json()}\n")
  token = f"Bearer {token_request.json()['access_token']}"

  log.debug(f"This is a valid token you can copy paste:\n{token}\n")
  return token


def get_credentials(name,config):
  """
  This function receives a server 'name' and 'config', checks 
  the options in the server configuration and gets the username 
  and password according to what is given:
  - if 'username' and 'password' are given, read them and return
  - if "credentials: 'module'" is chosen, then a module 'credentials' with a function 'get_user_pass' 
    must be in PYTHONPATH and "return username,password"
  - if "credentials: 'none'" is used, perform queries without authentication
  - if username and/or password are not obtained from the options above,
    ask in the command line (if 'keyring' module is present, store password there)
    - if no username is given, perform queries without authentication
  """
  log = logging.getLogger('logger')
  username = None
  password = None
  if "credentials" in config:
    if isinstance(config['credentials'],dict):
      # Trying to get 'username' and 'password' from configuration
      # password is only tried if username is present
      if ('username' not in config['credentials']):
        log.error("'username' not in credentials configuration! Skipping...\n")
      else:
        username = os.path.expandvars(config['credentials']['username'])
        if ('password' not in config['credentials']):
          log.warning("'password' not in credentials configuration! Skipping...\n")
        else:
          password = os.path.expandvars(config['credentials']['password'])
    elif config['credentials'] == 'module':
      try: 
        # Internal function
        from credentials import get_user_pass
        username,password = get_user_pass()
      except ModuleNotFoundError:
        log.critical("Credentials was chosen to be obtained via module, but module 'credentials' does not exist!\n")
    elif config['credentials'] == 'none':
      log.debug("Queries will be done without authentication\n")
      return None,None
  # If username was not obtained in config or module, ask now
  if not username:
    username = input("Username:")
    if not username:
      log.info("No username given, queries will be done without authentication\n")
      return None,None
  # If username was not obtained in config or module, ask now
  if not password:
    try:
      # Trying keychain first
      import keyring
      password = keyring.get_password('llview_prometheus', username)
      if password is None:
        getpass.getpass(f"Enter password for {username} on '{name}':")
        password = getpass.getpass(username,'llview_prometheus')
        keyring.set_password(name, username, password)
    except ImportError:
      log.error("Keyring module cannot be imported, password will not be saved.\n")
      password = getpass.getpass(f"Enter password for {username}:")
  return username,password


def parse_config_yaml(filename):
  """
  YML configuration parser
  """
  import yaml
  with open(filename, 'r') as configyml:
    configyml = yaml.safe_load(configyml)
  return configyml

class CustomFormatter(logging.Formatter):
  """
  Formatter to add colors to log output
  (adapted from https://stackoverflow.com/a/56944256/3142385)
  """
  def __init__(self,fmt,datefmt=""):
    super().__init__()
    self.fmt=fmt
    self.datefmt=datefmt
    # Colors
    self.grey = "\x1b[38;20m"
    self.yellow = "\x1b[93;20m"
    self.blue = "\x1b[94;20m"
    self.magenta = "\x1b[95;20m"
    self.cyan = "\x1b[96;20m"
    self.red = "\x1b[91;20m"
    self.bold_red = "\x1b[91;1m"
    self.reset = "\x1b[0m"
    # self.format = "%(asctime)s %(funcName)-18s(%(lineno)-3d): [%(levelname)-8s] %(message)s"

    self.FORMATS = {
                    logging.DEBUG: self.cyan + self.fmt + self.reset,
                    logging.INFO: self.grey + self.fmt + self.reset,
                    logging.WARNING: self.yellow + self.fmt + self.reset,
                    logging.ERROR: self.red + self.fmt + self.reset,
                    logging.CRITICAL: self.bold_red + self.fmt + self.reset
                  }
    
  def format(self, record):
    log_fmt = self.FORMATS.get(record.levelno)
    formatter = logging.Formatter(fmt=log_fmt,datefmt=self.datefmt)
    return formatter.format(record)
    
# Adapted from: https://stackoverflow.com/a/53257669/3142385
class _ExcludeErrorsFilter(logging.Filter):
    def filter(self, record):
        """Only lets through log messages with log level below ERROR ."""
        return record.levelno < logging.ERROR

log_config = {
              'format': "%(asctime)s %(funcName)-18s(%(lineno)-3d): [%(levelname)-8s] %(message)s",
              'datefmt': "%Y-%m-%d %H:%M:%S",
              # 'file': 'slurm.log',
              # 'filemode': "w",
              'level': "INFO" # Default value; Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR' from more to less verbose logging
              }
def log_init(level):
  """
  Initialize logger
  """

  # Getting logger
  log = logging.getLogger('logger')
  log.setLevel(level if level else log_config['level'])

  # Setup handler (stdout, stderr and file when configured)
  oh = logging.StreamHandler(sys.stdout)
  oh.setLevel(level if level else log_config['level'])
  oh.setFormatter(CustomFormatter(log_config['format'],datefmt=log_config['datefmt']))
  oh.addFilter(_ExcludeErrorsFilter())
  oh.terminator = ""
  log.addHandler(oh)  # add the handler to the logger so records from this process are handled

  eh = logging.StreamHandler(sys.stderr)
  eh.setLevel('ERROR')
  eh.setFormatter(CustomFormatter(log_config['format'],datefmt=log_config['datefmt']))
  eh.terminator = ""
  log.addHandler(eh)  # add the handler to the logger so records from this process are handled

  if 'file' in log_config:
    fh = logging.FileHandler(log_config['file'], mode=log_config['filemode'])
    fh.setLevel(level if level else log_config['level'])
    fh.setFormatter(CustomFormatter(log_config['format'],datefmt=log_config['datefmt']))
    fh.terminator = ""
    log.addHandler(fh)  # add the handler to the logger so records from this process are handled
  return

################################################################################
# MAIN PROGRAM:
################################################################################
def main():
  """
  Main program
  """
  
  # Parse arguments
  parser = argparse.ArgumentParser(description="Prometheus Plugin for LLview")
  parser.add_argument("--config",    default=False, help="YAML config file containing the information to be gathered and converted to LML")
  parser.add_argument("--loglevel",  default=False, help="Select log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR' (more to less verbose)")
  parser.add_argument("--singleLML", default=False, help="Merge all sections into a single LML file")
  parser.add_argument("--outfolder", default=False, help="Reference output folder")

  args = parser.parse_args()

  # Configuring the logger (level and format)
  log_init(args.loglevel)
  log = logging.getLogger('logger')

  if args.config:
    config = parse_config_yaml(args.config)
  else:
    log.critical("Config file not given!\n")
    parser.print_help()
    exit(1)

  if (args.singleLML):
    unique = Info()

  for servername,server_config in config.items():
    log.debug(f"Processing Server '{servername}'\n")

    # Checking if something is to be done on current server
    if ('files' not in server_config) or (len(server_config['files']) == 0):
      log.warning("No files to process on this server. Skipping...\n")
      continue
    if ('hostname' not in server_config):
      log.warning("No 'hostname' to collect on this server. Skipping...\n")
      continue

    # Getting credentials for the current server
    username, password = get_credentials(servername,server_config)

    token = None
    if 'token' in server_config and len(server_config['token']):
      token = get_token(username,password,server_config['token'])
      if not token:
        log.error(f"Token was defined but could not be obtained. Skipping server {servername}...\n")
        continue

    # Cached queries: to be used in different files without querying again
    cached_queries = {}

    #####################################################################################
    # Processing config file
    for filename,file in server_config['files'].items():
      if (not args.singleLML) and ('LML' not in file):
        log.error(f"No LML file given for {filename} in config! Skipping file...\n")
        continue

      start_time = time.time()

      log.info(f"Collecting {filename}...\n")

      # Initializing new object of type given in config
      info = Info(
                  hostname=server_config['hostname'],
                  username=username,
                  password=password,
                  token=token
                  )

      # Parsing Slurm output
      if ('metrics' not in file) or (len(file['metrics']) == 0):
        log.warning(f"No 'metrics' given for file {filename} in config file! Skipping...\n")
      else:
        cached_queries |= info.query(
                                      file['metrics'],
                                      prefix=file['prefix'] if 'prefix' in file else 'i',
                                      stype=file['type'] if 'type' in file else 'item',
                                      cached_queries = cached_queries,
                                    )

      # Modifying SLURM output with functions
      if 'modify_after_parse' in file:
        info.modify(file['modify_after_parse'])

      # Using function of name 'key' (current key being processed, e.g.: prometheus, etc.), when defined,
      # to modify that particular group/dictionary and items
      if filename in globals():
        func = globals()[filename]
        info.add(func(file,info))

      # Modifying SLURM output with functions
      if 'modify_before_mapping' in file:
        info.modify(file['modify_before_mapping'])

      # Applying pattern to include or exclude units
      if 'exclude' in file or 'include' in file:
        info.apply_pattern(
                            exclude=file['exclude'] if 'exclude' in file else '',
                            include=file['include'] if 'include' in file else ''
                          )

      # Mapping keywords
      if 'mapping' in file:
        info.map(file['mapping'])

      end_time = time.time()
      log.debug(f"Gathering {filename} information took {end_time - start_time:.4f}s\n")

      # Add timing key
      # if not info.empty():
      timing = {}
      name = f'get{filename}'
      timing[name] = {}
      timing[name]['startts'] = start_time
      timing[name]['datats'] = start_time
      timing[name]['endts'] = end_time
      timing[name]['duration'] = end_time - start_time
      timing[name]['nelems'] = len(info)
      # The __nelems_{type} is used to indicate to DBupdate the number of elements - important when the file is empty
      timing[name][f"__nelems_{file['type'] if 'type' in file else 'item'}"] = len(info)
      timing[name]['__type'] = 'pstat'
      timing[name]['__id'] = f'pstat_get{filename}'
      info.add(timing)

      if (not args.singleLML):
        if info.empty():
          log.warning(f"Object {filename} is empty, nothing to output to LML! Skipping...\n")
        else:
          info.to_LML(f"{args.outfolder+'/' if args.outfolder else ''}{file['LML']}")
      else:
        # Accumulating for a single LML
        unique = unique + info

  if (args.singleLML):
    if unique.empty():
      log.warning(f"Unique object is empty, nothing to output to LML! Skipping...\n")
    else:
      unique.to_LML(f"{args.outfolder+'/' if args.outfolder else ''}{args.singleLML}")

  log.debug("FINISH\n")
  return

if __name__ == "__main__":
  main()
