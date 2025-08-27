#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
# and Camp project contributors. See the camp/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#=============================================================================
# Change the copyright date in all files that contain the text
# "the camp/LICENSE file", which is part of the copyright statement 
# at the top of each Camp file. We use this to distinguish Camp files from 
# that we do not own (e.g., other repos included as submodules), which we do
# not want to modify. Note that this file and *.git files are omitted
# as well.
#
# IMPORTANT: Since this file is not modified (it is running the shell 
# script commands), you must EDIT THE COPYRIGHT DATES IN THE HEADER ABOVE 
# MANUALLY.
#
# Edit the 'find' command below to change the set of files that will be
# modified.
#
# Change the 'sed' command below to change the content that is changed
# in each file and what it is changed to.
#
#=============================================================================
#
# If you need to modify this script, you may want to run each of these 
# commands individually from the command line to make sure things are doing 
# what you think they should be doing. This is why they are separated into 
# steps here.
# 
#=============================================================================

#=============================================================================
# First find all the files we want to modify
#=============================================================================
find . -type f ! -name \*.git\*  ! -name \*update_copyright\* -exec grep -l "the camp/LICENSE file" {} \; > files2change

#=============================================================================
# Replace the old copyright dates with new dates
#=============================================================================
for i in `cat files2change`
do
    echo $i
    cp $i $i.sed.bak
    sed "s/Copyright (c) 2018-24/Copyright (c) 2018-25/" $i.sed.bak > $i
done

echo LICENSE
cp LICENSE LICENSE.sed.bak
sed "s/Copyright (c) 2018-2024/Copyright (c) 2018-2025/" LICENSE.sed.bak > LICENSE

for i in README.md docs/conf.py
do 
    echo $i
    cp $i $i.sed.bak
    sed "s/2018-24/2018-25/" $i.sed.bak > $i
done

#=============================================================================
# Remove temporary files created in the process
#=============================================================================
find . -name \*.sed.bak -exec rm {} \;
rm files2change
