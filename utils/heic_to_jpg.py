import os, subprocess

directory = '../samples'

for filename in os.listdir(directory):
    if filename.lower().endswith(".jpeg"):
        print('Converting %s...' % os.path.join(directory, filename))
        subprocess.run([
            "magick",
            "%s" % os.path.join(directory, filename),
            "%s" % (os.path.join(directory, filename[0:-5]) + '.jpg')
        ])
        continue
