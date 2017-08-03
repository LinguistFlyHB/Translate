import urllib.request
from imp import reload
import sys
from bs4 import BeautifulSoup

reload(sys)

fp = open('F:/a.txt', 'a')
start = 0
while start <= 1500:
    link = "http://www.imdb.com/title/tt0111161/reviews?start="+str(start)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    }
    response = urllib.request.Request(link, headers=headers)
    content = urllib.request.urlopen(response)
    soup = BeautifulSoup(content, 'lxml').find('div', {'id': "tn15content"})
    div = soup.find_all('div')
    comment = soup.find_all('p')
    print(len(div))
    print(len(comment))
    k = 0
    for i in range(len(comment)):
        if comment[i].find('b') is not None or comment[i].find('a') is not None:
            continue
        if len(div[k].find_all('img')) != 2:
            k = k + 2
            continue
        judge = div[k].find_all('img')[1]['alt']
        k = k + 2
        try:
            fen = judge.replace('/10', '')
            pin = str(comment[i]).replace('<p>', '').replace('</p>', '').replace("<br/><br/>", '').replace('\n', '')
            print(pin)
            fp.write(fen)
            fp.write("\n")
            fp.write(pin)
            fp.write("\n\n\n")
        except UnicodeEncodeError:
            print('UnicodeEncodeError')
    start = start + 10
    print(start)
fp.close()
