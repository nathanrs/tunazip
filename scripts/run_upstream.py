import sys

sys.path.insert(0, sys.argv[1])
import nn as upstream

upstream.V2_CONFIG["distributed"] = False
sys.argv = ["nn.py", *sys.argv[2:]]
sys.exit(upstream.main())
