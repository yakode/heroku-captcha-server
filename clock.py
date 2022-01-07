from apscheduler.schedulers.blocking import BlockingScheduler
import urllib.request as urllib

sched_self = BlockingScheduler()

@sched_self.scheduled_job('cron', hour='0-12', minute='*/20')
def scheduled_job():
    url = "https://eeclass-captcha.herokuapp.com/"
    conn = urllib.urlopen(url)

sched_self.start()