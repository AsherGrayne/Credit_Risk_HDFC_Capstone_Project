#!/bin/bash
# Netlify Build Script
# This script prevents Python detection and just publishes the static site

# Remove runtime.txt temporarily to prevent Netlify from detecting Python
if [ -f "runtime.txt" ]; then
    mv runtime.txt runtime.txt.backup
fi

# No build needed - this is a static site
echo "Building static site - no dependencies needed"

# Restore runtime.txt (though it won't be in the published site)
if [ -f "runtime.txt.backup" ]; then
    mv runtime.txt.backup runtime.txt
fi

# Exit successfully
exit 0

