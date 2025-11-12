.. # Copyright (c) 2018-2025, Lawrence Livermore National Security, LLC and
.. # other Camp project contributors. See the camp/LICENSE file for details.
.. # 
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _camp-dev-guide-label:

####################
Camp Developer Guide
####################

The Camp developer guide is here to help new Camp developers develop for Camp. 

Contributing to Camp
====================

Camp is a collaborative open-source software project and we encourage contributions from anyone 
who wants to add features or improve its capabilities. Camp is developed in conjunction with the
`RAJA <https://github.com/LLNL/RAJA>`_ and `Umpire <https://github.com/LLNL/Umpire>`_ code libraries since Camp is a part of the RAJA Portability Suite.

We assume contributors are familiar with 
`Git <https://git-scm.com/>`_, which we use for source code version control,
and `GitHub <https://github.com/>`_, which is where our project is hosted. 

GitHub Project Access
---------------------

Camp maintains levels of project access on its GitHub project:

  * **Core team members.** Individuals on the core Camp and RAJA team are frequent
    Camp contributors and participate regularly in project meetings, 
    discussions, and other project activities. Their
    project privileges include the ability to create branches in the repository,
    push code changes to the Camp repo, make PRs, and merge them when they are 
    approved and all checks have passed.
  * **Everyone else.** Anyone with a GitHub account is welcome to contribute
    to Camp. Individuals outside of the group described above can make PRs
    in the Camp project, but must do so from a branch on a *fork* of 
    the Camp repo. 

If you need access to the Camp repo, email raja-dev@llnl.gov

Before a PR can be merged into Camp, all test checks must pass and the PR must be approved 
by at least one member of the core Camp or RAJA team.

Each Camp contribution (feature, bugfix, etc.) must include adequate tests, documentation, 
and code examples. The adequacy of PR content, in this respect, is determined by PR reviewers 
applying their professional judgment considering the perspective of RAJA, Camp, and Umpire users and developers.

Release Process
===============

.. note:: No significant code development is performed on a release branch.
          In addition to preparing release notes and other documentation, the
          only code changes that should be done are bug fixes identified
          during release preparations

Here are the steps to follow when creating a Camp release.

1: Start Release Candidate Branch
---------------------------------

Create a release candidate branch off of the develop branch to initiate a
release. The name of a release branch must contain the associated release version
number. Typically, we use a name like v2024.07 where 2024 corresponds to the year
and 07 corresponds to the month in which the release was made. 

.. important:: 
   Releases are coordinated with the RAJA and Umpire teams as part of timely
   RAJA Portability Suite releases. The release names will correspond to the release
   names of RAJA and Umpire as part of this process. Be sure to coordinate any Camp
   releases and work with the RAJA and Umpire teams for a release schedule.

For example:

.. code:: bash

    git checkout -b v2024.07

2: Update Versions in Code
--------------------------

Update the version of the code where ever it is documented (i.e. the README file, Doxygen, etc.)
and make sure the new version numbers are consistent.

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

#. Enter the information for the release into the release description.

#. Publish the release. This will add a corresponding entry in the
   `Releases section <https://github.com/LLNL/camp/releases>`_

#. Communicate with the RAJA team when this process is done to properly coordinate releases.

.. note::

   Github will add a corresponding tarball and zip archives consisting of the
   source files for each release.


5: Create Release Branch and Mergeback to main
-------------------------------------------------

1. Create a branch off main that is for the release branch.

.. code:: bash

    git pull
    git checkout main
    git checkout -b release-v2024.07
    git push --set-upstream origin release-v2024.07


2. Create a pull request to merge into ``main``. When approved, merge it.

If you have questions regarding this process, reach out to Camp developers or
send an email to raja-dev@llnl.gov
