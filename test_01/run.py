from threading import Thread
import pytest
import json
import time

config = json.load(open(r"D:\tagging\script\test_01\config.json"))


class VAR(dict):
    def __getattr__(self, key):
        if key not in self:
            return None
        else:
            value = self[key]
            if isinstance(value, dict):
                return VAR(value)
            return value


var = VAR(config)



def doing(sn):
    pytest.main(["--cmdopt=%s" % sn, "--html=./report/%s.html" % ("%.2f"%time.time()+ sn), "-s"])

for i in range(10):
    threads = []
    for i in ["fdsghbnb", "dfrgtbvcds"]:
        threads.append(Thread(target=doing, args=(i,)))

    for i in threads:
        i.start()

    for i in threads:
        i.join()
