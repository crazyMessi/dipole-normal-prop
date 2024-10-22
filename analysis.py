import os
import json
import field_utils
import util


def get_segs(floder,seg_type="res"):
    class Segment:
        def __init__(self, id) -> None:
            self.id = id
            filename = seg_type+str(id)+".ply"
            trans, nxyz = util.load_and_trans_tensor(floder + filename)
            self.nxyz = nxyz
            self.trans = trans
            log_i = [x for x in log if x["id"] == id]
            if len(log_i) == 0:
                self.metric = None
                return
            else:
                log_i = log_i[0]
            self.metric = log_i["metric"]
    json_path = floder + "reslog.json"
    log = json.load(open(json_path, "r"))
    log = log["node_log"]
    filelist = os.listdir(floder)
    filelist = [x for x in filelist if seg_type in x]
    filelist = [x for x in filelist if x.endswith(".ply")]    
    segs = [Segment(i) for i in range(len(filelist))]
    segs = [s for s in segs if s.metric is not None]
    return segs


