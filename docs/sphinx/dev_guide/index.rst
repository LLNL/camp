.. # Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
.. # other BLT Project Developers. See the top-level LICENSE file for details
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

####################
CAMP Developer Guide
####################

The camp developer guide is here to help new CAMP developers develop for camp. 

Release Process
===============

.. note:: No significant code development is performed on a release branch.
          In addition to preparing release notes and other documentation, the
          only code changes that should be done are bug fixes identified
          during release preparations

Here are the steps to follow when creating a CAMP release.

1: Start Release Candidate Branch
---------------------------------

Create a release candidate branch off of the develop branch to initiate a
release. The name of a release branch must contain the associated release version
number. Typically, we use a name like v2024.07 where 2024 corresponds to the year
and 07 corresponds to the month in which the release was made. Releases are
coordinated with the RAJA and Umpire teams.

.. code:: bash

    git checkout -b v2024.07

2: Update Versions in Code
--------------------------

**Update Release Notes**

Update any notes for a new release and update the license disclaimers at the top of any files
if applicable.

3: Create Pull Request and push a git `tag` for the release
-----------------------------------------------------------

#. Commit the changes and push them to Github.
#. Create a pull request from release candidate branch to ``main`` branch.
#. Merge pull request after reviewed and passing tests.
#. Checkout main locally: ``git checkout main && git pull``
#. Create release tag:  ``git tag v2024.07``
#. Push tag to Github: ``git push --tags``


4: Draft a Github Release
-------------------------

`Draft a new Release on Github <https://github.com/LLNL/camp/releases/new>`_

#. Enter the desired tag version, e.g., *v2024.07*

#. Select **main** as the target branch to tag a release.

#. Enter a Release title with the same as the tag *v2024.07*

#. Enter the information for the release into the release description (omit any sections if empty).

#. Publish the release. This will add a corresponding entry in the
   `Releases section <https://github.com/LLNL/camp/releases>`_

.. note::

   Github will add a corresponding tarball and zip archives consisting of the
   source files for each release.


5: Create Release Branch and Mergeback to develop
-------------------------------------------------

1. Create a branch off main that is for the release branch.

.. code:: bash

    git pull
    git checkout main
    git checkout -b release-v2024.07
    git push --set-upstream origin release-v2024.07


2. Create a pull request to merge into ``main``. When approved, merge it.

