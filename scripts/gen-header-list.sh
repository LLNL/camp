#!/bin/sh
echo "set(camp_headers"
find include -name '*.hpp' | grep -v '\.in\.hpp'
echo ")"
