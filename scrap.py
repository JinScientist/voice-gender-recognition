import sys, urllib, re, urlparse
from urllib import urlretrieve
from BeautifulSoup import BeautifulSoup

datapath='./rawdata/'
url='http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'
f = urllib.urlopen(url)
soup = BeautifulSoup(f)
for i in soup.findAll('a',attrs={'href': re.compile('(?i)(tgz)$')}):
	full_url = urlparse.urljoin(url, i['href'])
	print full_url
	response=urllib.urlopen(full_url)
	tgzfile=urllib.urlopen(full_url).read()
	urlretrieve(full_url, datapath+i['href'])