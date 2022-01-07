from apscheduler.schedulers.blocking import BlockingScheduler
import urllib.request as urllib

sched_self = BlockingScheduler()
sched_other = BlockingScheduler()

@sched_self.scheduled_job_self('cron', hour='0-12', minute='*/20')
def scheduled_job_self():
    url = "https://eeclass-captcha.herokuapp.com/"
    conn = urllib.urlopen(url)

@sched_other.scheduled_job_other('cron', hour='9', minute='25')
def scheduled_job_other():
    url = "https://elearn-captcha.herokuapp.com/"
    conn = urllib.urlopen(url)


sched_self.start()
sched_other.start()