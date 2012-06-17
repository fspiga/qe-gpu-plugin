# Copyright (C) 2001-2006 Quantum ESPRESSO group
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Author: Filippo Spiga (spiga.filippo@gmail.com)
# Date: June 14, 2012
# Version: 1.1

import os
import os.path
import sys
import shutil

s=raw_input()
while s.strip()!='':
	input_file = os.path.abspath(s)
	backup_file = os.path.join(os.path.dirname(input_file), '.' + os.path.basename(input_file) + '.orig')
	
        print input_file+": ",
	
	if os.path.exists(backup_file) :
		# restore original
		os.rename(backup_file, input_file) 
		
		print "restored."
	else:
		print "skipped (do a manual check)."
	
	try:
		s=raw_input()
	except:
		exit()
