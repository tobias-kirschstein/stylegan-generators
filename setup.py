#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    # Still necessary, otherwise we get a pip error
    setuptools.setup(include_package_data=True,
                     # package_data={'': ['*.cpp']},
                     )
