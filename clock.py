from apscheduler.schedulers.blocking import BlockingScheduler
import urllib.request as urllib
import time

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour='0-12', minute='*/1')
def scheduled_job():
    url = "https://eeclass-captcha.herokuapp.com/"
    conn = urllib.urlopen(url)
    now = time.gmtime(time.time())
    print("=================================")
    print(now)
    print("=================================")
    if now.tm_hour == 2:
        if now.tm_min >= 0:
            print("ya")
            url = "https://elearn-captcha.herokuapp.com/"
            conn = urllib.urlopen(url)

sched.start()