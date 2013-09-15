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
import hashlib
import json

# Control variables
json_loaded = False
perform_replace = False
dump_json = False

# Load JSON
gemm_input_file = "/tmp/GEMM_files.json"
if os.path.exists(gemm_input_file) :
	file_gemmed = open(gemm_input_file)
	gemmed_data = json.load(file_gemmed)
	json_loaded = True
else :
	gemmed_data = {}
	dump_json = True

# Read stdin and perform symbols replacement
raw_input = sys.stdin.readlines()
for line in raw_input:
	input_file = os.path.abspath(line.rstrip())
	computed_hash = hashlib.md5( open( input_file , 'r').read()).hexdigest()
	
	# DEBUG # print input_file+": hashed"
	
	if json_loaded :
		# JSON loaded but file missing --> add new file and dump the JSON
		if input_file in gemmed_data :
			# Trigger symbol replacement _only_ if hash mismatch
			if (computed_hash != gemmed_data[input_file]) :
				perform_replace = True
			else :
				print input_file+": skipped."		
		else:
			dump_json = True
			gemmed_data[input_file] = computed_hash
			print input_file+": re-hashed."		
	else : 
		gemmed_data[input_file] = computed_hash
		print input_file+": hashed."
	
	# Perform symbol replacement
	if perform_replace :
		backup_file = os.path.join(os.path.dirname(input_file), '.' + os.path.basename(input_file) + '.orig')
		tmp_file = os.path.join(os.path.dirname(input_file), '.' + os.path.basename(input_file)+'~')
		
		print input_file+": ",
		
		if os.path.exists(backup_file) :
			print "skipped."
		else:
			# read the current contents of the file 
			f = open( input_file )
			text = f.read() 
			f.close() 
			
			# backup original file
			shutil.copy (input_file, backup_file)
			
			# open a different file for writing
			f = open(tmp_file, 'w') 
			f.write("""
#if defined(__PHIGEMM)
#define dgemm UDGEMM  
#define zgemm UZGEMM  
#define DGEMM UDGEMM  
#define ZGEMM UZGEMM  
#if defined(__PHIGEMM_PROFILE)
#define _STRING_LINE_(s) #s
#define _STRING_LINE2_(s) _STRING_LINE_(s)
#define __LINESTR__ _STRING_LINE2_(__LINE__)
#define UDGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC) phidgemm(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC,__FILE__,__LINESTR__)
#define UZGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC) phizgemm(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC,__FILE__,__LINESTR__)
#else
#define UDGEMM phidgemm
#define UZGEMM phizgemm
#endif
#endif
			""")
			# Write the original contents 
			f.write(text) 
			f.close() 
			
			# Overwrite
			os.rename(tmp_file, input_file) 
			
			print "success."

# Dump JSON
if dump_json :
	json_string = json.dumps(gemmed_data)
	# DEBUG # print str(json_string)
	output_json = open("/tmp/GEMM_files.json", 'wb')
	json.dump(gemmed_data, output_json)
	output_json.close

# Necessary?
sys.exit()
