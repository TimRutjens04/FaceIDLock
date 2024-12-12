import RPi.GPIO as GPIO
import time
import logging

# Logging setup:
logging.basicConfig(
    filename="door_lock.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Servo setup:
SERVO_PIN = 11  # Change this to the appropriate pin number for your setup

# Set pin numbering mode (BOARD or BCM)
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
# Alternatively, you can use GPIO.setmode(GPIO.BCM) for Broadcom numbering

GPIO.setup(SERVO_PIN, GPIO.OUT)
p = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz for standard servo motors
p.start(0)

def set_servo_angle(angle):
    """Sets the servo to a specific angle."""
    duty_cycle = 2 + (angle / 18)  # Maps angle (0-180) to duty cycle (2-12)
    GPIO.output(SERVO_PIN, True)
    p.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    p.ChangeDutyCycle(0)

def unlock_door():
    """Unlocks the door by moving the servo to 90 degrees."""
    logging.info("Unlocking door...")
    set_servo_angle(90)

def lock_door():
    """Locks the door by moving the servo to 0 degrees."""
    logging.info("Locking door...")
    set_servo_angle(0)

# Clean up GPIO when the program exits
def cleanup_gpio():
    logging.info("Cleaning up GPIO...")
    p.stop()
    GPIO.cleanup()

# Example usage
if __name__ == "__main__":
    try:
        unlock_door()
        time.sleep(2)  # Keep the door unlocked for 2 seconds
        lock_door()
    finally:
        cleanup_gpio()
