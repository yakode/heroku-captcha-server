from apscheduler.schedulers.blocking import BlockingScheduler
import urllib.request as urllib

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour='0-9', minute='*/2')
def scheduled_job():
    url = "https://eeclass-captcha.herokuapp.com/"
    conn = urllib.urlopen(url)


sched.start()