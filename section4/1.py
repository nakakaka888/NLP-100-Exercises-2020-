import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

font_path = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams["font.family"] = font_prop.get_name()

 
matplotlib.use('Agg') # -----(1)

plt.plot([1,2,3,4,5],[1,2,3,4,5], "bx-", label="1次関数")
plt.plot([1,2,3,4,5],[1,4,9,16,25],"ro--", label="2次関数")

plt.xlabel("xの値")
plt.ylabel("yの値")

plt.legend(loc="best")
plt.xlim(0, 6)
plt.savefig("japanese_label.png", dpi=300)