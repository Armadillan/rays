import os
import shutil

missing = ['285627.578369', '285473.078369', '286128.578369', '285332.078369', '285593.078369', '285575.078369', '285299.078369', '285096.578369', '285342.578369', '285312.578369', '286188.578369', '286074.578369', '285362.078369', '285078.578369', '285666.578369', '286047.578369', '285770.078369', '285752.078369', '285641.078369', '286253.078369', '285425.078369', '285123.578369', '285993.578369', '286313.078369', '285470.078369', '286089.578369', '285095.078369', '286295.078369', '285396.578369', '285204.578369', '285917.078369', '285533.078369', '286290.578369', '286278.578369', '286149.578369', '285818.078369', '286311.578369', '285813.578369', '285261.578369', '285167.078369', '285953.078369', '285560.078369', '285485.078369', '285237.578369', '286268.078369', '285192.578369', '285717.578369', '285477.578369', '285842.078369', '286118.078369', '285681.578369', '285612.578369', '286310.078369', '285410.078369', '285354.578369', '285855.578369', '285150.578369', '286236.578369', '285608.078369', '285554.078369', '285386.078369', '285456.578369', '286007.078369', '285716.078369', '285563.078369', '285447.578369', '286185.578369', '285174.578369', '285459.578369', '285393.578369', '285645.578369', '285483.578369', '285671.078369', '285969.578369', '285350.078369', '285251.078369', '285194.078369', '286257.578369', '286230.578369', '285240.578369', '285731.078369', '286209.578369', '285812.078369', '285131.078369', '285858.578369', '285602.078369', '285986.078369', '285245.078369', '286038.578369', '285932.078369', '286218.578369', '285494.078369', '285606.578369', '285920.078369', '285578.078369', '285039.578369', '285462.578369', '286217.078369', '285173.078369', '285885.578369', '285375.578369', '285914.078369', '285543.578369', '285762.578369', '285084.578369', '286299.578369', '285935.078369', '286227.578369', '285083.078369', '285567.578369', '285509.078369', '286008.578369', '285879.578369', '285711.578369', '285231.578369', '285911.078369', '285975.578369', '285809.078369', '285374.078369', '285404.078369', '285141.578369', '285048.578369', '285959.078369', '286122.578369', '285432.578369', '286134.578369', '286086.578369', '285270.578369', '286302.578369', '285735.578369', '285566.078369', '286160.078369', '286073.078369', '285333.578369', '285291.578369', '285614.078369', '285144.578369', '285116.078369', '285479.078369', '286272.578369', '285053.078369', '285146.078369', '286181.078369', '285822.578369', '286175.078369', '285695.078369', '286146.578369', '285183.578369', '285720.578369', '285060.578369', '285059.078369', '285047.078369', '285246.578369', '285750.578369', '285647.078369', '285588.578369', '285815.078369', '286182.578369', '285912.578369', '286284.578369', '285203.078369', '286040.078369', '285564.578369', '285213.578369', '285684.578369', '286251.578369']

computed = os.listdir("embeddings")

os.makedirs("found_embeddings", exist_ok=True)

for snapshot in missing:
    if snapshot + ".npy" in computed:
        print("copying")
        shutil.copyfile(
            "embeddings/" + snapshot + ".npy",
            "found_embeddings/" + snapshot + ".npy"
            )