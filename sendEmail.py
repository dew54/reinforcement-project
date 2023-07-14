import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders

def send_email(subject, message, sender, recipients, smtp_server, smtp_port, username, password, attachments=None):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)

    # Add the message body
    msg.attach(MIMEText(message, 'plain'))

    # Attach any files
    if attachments:
        for attachment in attachments:
            part = MIMEBase('application', 'octet-stream')
            with open(attachment, 'rb') as file:
                part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{attachment}"')
            msg.attach(part)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.sendmail(sender, recipients, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except smtplib.SMTPException as e:
        print("Error: Unable to send email.")
        print(e)

# Set up your email details
# subject = "Hello from Python!"
# message = "This is the body of the email."
# sender = "your_email@example.com"
# recipients = ["recipient1@example.com", "recipient2@example.com"]
# smtp_server = "smtp.example.com"
# smtp_port = 587
# username = "your_username"
# password = "your_password"
# attachments = ["file1.txt", "file2.pdf"]  # Replace with actual file paths

# # Call the send_email function to send the email with attachments
# send_email(subject, message, sender, recipients, smtp_server, smtp_port, username, password, attachments)
