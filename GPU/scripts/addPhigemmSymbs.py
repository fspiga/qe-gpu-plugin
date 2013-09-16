# Copyright (C) 2001-2006 Quantum ESPRESSO group
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Author: Filippo Spiga
# Date: September 15, 2013
# Version: 1.5

import os
import os.path
import sys
import shutil

# Read stdin and perform symbols replacement
raw_input = sys.stdin.readlines()
for line in raw_input:
	input_file = os.path.abspath(line.rstrip())
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

# Necessary?
sys.exit()
