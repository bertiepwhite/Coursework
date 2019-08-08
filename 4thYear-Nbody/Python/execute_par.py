import sys
from Gravity_par import main       #WATCH OUT HERE
from inspect import signature

if int(len(sys.argv)) == 2:

    main(int(sys.argv[1]))
else:
    print("Args incorrect")
