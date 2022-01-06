from apscheduler.schedulers.blocking import BlockingScheduler
import urllib

# 宣告一個排程
sched = BlockingScheduler()

# 定義排程 : 在周一至周五，每 20 分鐘就做一次 def scheduled_jog()
@sched.scheduled_job('cron', day_of_week='mon-sun', minute='*/29')
def scheduled_job():
    url = "https://eeclass-captcha.herokuapp.com/"
    connect = urllib.request.urlopen(url)
    
sched.start()  # 啟動排程