#!/usr/bin/python

# Copyright (C) 2001-2006 Quantum ESPRESSO group
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Author: Filippo Spiga (spiga.filippo@gmail.com)
# Date: December 13 , 2012
# Version: 0.1

import hashlib
import simplejson as json
import os, os.path

json_data=open("./GPU/scripts/qe-gpu-5.0.2.json")
data = json.load(json_data)

filenames = data['filenames']

for file in filenames.keys():
     file_to_check = os.path.abspath(filenames[file]['original'])
     hash_original = hashlib.md5( open( file_to_check , 'r').read()).hexdigest()
     if (hash_original != filenames[file]['hash']):
	  print "The corresponding GPU version of " + file_to_check + " is *NOT* aligned, please manually check."
          #print os.path.basename(file_to_check) + " is *NOT* aligned, please manually check."
