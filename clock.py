from apscheduler.schedulers.blocking import BlockingScheduler
import urllib.request as urllib
import time

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour='12-23', minute='*/15')
def scheduled_job():
    now = time.gmtime(time.time())
    if now.tm_hour == 23  and now.tm_min >= 45:
        print("wake up the other app")
        url = "https://eeclass-captcha.herokuapp.com/"
        conn = urllib.urlopen(url)
    else:
        print("wake up this app")
        url = "https://elearn-captcha.herokuapp.com/"
        conn = urllib.urlopen(url)

sched.start()