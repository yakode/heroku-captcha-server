from apscheduler.schedulers.blocking import BlockingScheduler
import urllib

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour='0-9', minute='*/2')
def scheduled_job():
    url = "https://eeclass-captcha.herokuapp.com/"
    conn = urllib.request.urlopen(url)


sched.start()