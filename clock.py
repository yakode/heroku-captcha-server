from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour='0-12', minute='*/29')
def scheduled_job():
    url = "https://eeclass-captcha.herokuapp.com/"
    conn = urllib.request.urlopen(url)
        
    # for key, value in conn.getheaders():
    #     print(key, value)

sched.start()