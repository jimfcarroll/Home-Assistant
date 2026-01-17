#!/bin/bash -x

if [ "$1" == "" ]; then
    echo "Usage: You should have passed the user:group"
    exit 1
fi

rm -rf package-lock.json
rm -rf node_modules
rm -rf storage

npm install
mkdir -p storage

chmod a+w storage
chown -R $1 package-lock.json node_modules storage


