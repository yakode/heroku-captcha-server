from apscheduler.schedulers.blocking import BlockingScheduler
import urllib.request as urllib
import time

sched_self = BlockingScheduler()

@sched_self.scheduled_job('cron', hour='0-12', minute='*/20')
def scheduled_job():
    url = "https://eeclass-captcha.herokuapp.com/"
    conn = urllib.urlopen(url)
    now = time.gmtime(1569376996)
    if now.tm_hour == 2:
        if now.tm_min >= 0:
            url = "https://elearn-captcha.herokuapp.com/"
            conn = urllib.urlopen(url)

sched_self.start()