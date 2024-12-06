import RPi.GPIO as GPIO
import time
import logging

#Logging setup:
logging.basicConfig(
    filename="door_lock.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#Servo setup:
SERVO_PIN = 11
GPIO.setup(SERVO_PIN, GPIO.OUT)
p = GPIO.PWM(SERVO_PIN, 50)
p.start(0)

def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18) #maps angle (0-180) to duty cycle (2-12)
    GPIO.output(SERVO_PIN, True)
    p.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    p.ChangeDutyCycle(0)

def unlock_door():
    logging.info("Unlocking door...")
    set_servo_angle(90)

def lock_door():
    logging.info("Locking door...")
    set_servo_angle(0)


