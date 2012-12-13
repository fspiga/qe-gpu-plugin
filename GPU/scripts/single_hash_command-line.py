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

import sys
import hashlib

u = hashlib.md5(open(sys.argv[1], 'r').read()).hexdigest()
print u
