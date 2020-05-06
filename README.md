### What statistcs are best-suited for predicting the outcome of a basketball game?

I'm a [Golden State Warriors](https://www.nba.com/warriors/) fan so I wanted to try to predict the outcomes of some of their games. This repo uses 2018-2019 [Klay Thompson](https://en.wikipedia.org/wiki/Klay_Thompson) [statistics](https://www.basketball-reference.com/players/t/thompkl01/gamelog/2019/) and [Sci-kit Learn](https://scikit-learn.org/) [decision trees](http://scikit-learn.org/stable/modules/tree.html) and [Pandas](https://pandas.pydata.org/) in Python, but you can use other data as well.

This post uses Twilio SMS to provide user input into the different statistics (offensive rebounds, field goals, 3-pointers, etc) used to try to predict if the Warriors won or not. You'll need 
- A Twilio account - [sign up for a free one here and receive an extra $10 if you upgrade through this link](http://www.twilio.com/referral/iHsJ5D)
- A Twilio phone number with SMS capabilities - [configure one here](https://www.twilio.com/console/phone-numbers/search)
- [Set up your Python and Flask developer environment](https://www.twilio.com/docs/usage/tutorials/how-to-set-up-your-python-and-flask-development-environment). Make sure you have [Python 3](https://www.python.org/downloads/) downloaded as well as [ngrok](https://ngrok.com/).

Clone the repo and run `pip3 install sklearn pandas numpy flask twilio`. Your Flask app will need to be visible from the web so Twilio can send requests to it. Ngrok lets us do this. With Ngrok installed, run the following command in your terminal in the directory this code is in. Run `ngrok http 5000` in a new terminal tab. Take that forwarding https:// URL ![ngrok url](https://lh5.googleusercontent.com/b8JNoLA720-HP9Q6m7fHBViQN6q_MAalxTglR2_myUIJFHgC3XZuQR2khwsLWRSv5dXx16Orz_pS02wpwz4iTPFDTj23fMGAMhUYZ6sorWcaP2LbiSrlcKdujEBM_D4N91nDJ34P)

to buy or configure an existing Twilio number in [your phone number console](https://www.twilio.com/console/phone-numbers/incoming). To buy one, search for a phone number in your country and region, making sure the SMS checkbox is ticked.

![phone number buying](https://lh5.googleusercontent.com/6b3bw6SVFhwQxPOwoiBcYLC62fK1dMu-luciDexWoZFhoXD8DYLsAKqPAMF-lx5QSF14sKEAa_XsFTKOMdCLMEJGoWrlIK_24i7S-r-JBNpnEhohE2Tm77pjAGLcL285mCnkkZl-)

In the `Messaging` section of your purchased number, in the `A Message Comes In`section, set the `Webhook` to be your ngrok https URL, appending `/sms` to the URL. Hit Save.

![messaging webhook console](https://lh6.googleusercontent.com/EG1MeVp4gYTCwSSZF6pX2YpLK4NZHHH5lJvofzSygqCsW-Kb4sjWc1_7Kr8A4xG0_EESkW65l1sYm1wptlzMpLA03ryP4VTDKEHAgbx1IfdoJ4girSX0M31X10qDO1i9DHCnE6yP)

If you run 
`export FLASK_APP=predict_with_klay
export FLASK_ENV=development
flask run` 
on Mac (if you're on Windows, replace `export` with `set`) from the command line and text your Twilio number a statistic, you should get a response like this:
![sms example](https://lh3.googleusercontent.com/z3-K8ITmiC-Wdx1KDLKW9OMeSyQQqt-aNfYj6Z5V_krKSeJbexb6GASFCIsq22TB9DQGrq4RQfSWCrcw2VQQ0TzfMXClpLF6ACr-CXWXmvaCdAV2uecWV4u_lIoeJOlUsNJXEqxW)