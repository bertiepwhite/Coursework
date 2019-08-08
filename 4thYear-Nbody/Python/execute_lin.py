import sys
from Gravity_lin import main
from inspect import signature

if int(len(sys.argv)) == 1:
    main()
else:
    print("Args incorrect")
