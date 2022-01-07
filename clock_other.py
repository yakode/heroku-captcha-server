from apscheduler.schedulers.blocking import BlockingScheduler
import urllib.request as urllib

sched_other = BlockingScheduler()

@sched_other.scheduled_job('cron', hour='0-12', minute='*/2')
def scheduled_job():
    url = "https://elearn-captcha.herokuapp.com/"
    conn = urllib.urlopen(url)

sched_other.start()