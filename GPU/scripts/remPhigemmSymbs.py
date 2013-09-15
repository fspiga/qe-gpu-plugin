# Copyright (C) 2001-2006 Quantum ESPRESSO group
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Author: Filippo Spiga
# Date: September 14, 2013
# Version: 1.4

import os
import os.path
import sys
import shutil

raw_input = sys.stdin.readlines()
for line in raw_input:
	input_file = os.path.abspath(line.rstrip())
	backup_file = os.path.join(os.path.dirname(input_file), '.' + os.path.basename(input_file) + '.orig')
	
	if os.path.exists(backup_file) :
		# restore original
		os.rename(backup_file, input_file) 
	#else:
		# DEBUG # print input_file + ": skipped (do a manual check)."
	
sys.exit()