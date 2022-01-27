import neptune.new as neptune

def graph(avgrewards, rewards, policy_loss, value_loss,  projectname):


    run = neptune.init(
    project=projectname,
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYTIzZmNmNS04YjRiLTQyMWMtYmIzMy1kOGEwNTE0NmRjOWQifQ==",
)  # your credentials

    for reward in rewards:
        run["rewards per episode"].log(reward)
    for reward in avgrewards:
        run["AVG BATCH REWARD"].log(reward)
    for loss in policy_loss:
        run["POLICY_NET LOSS"].log(loss)
    for loss in value_loss:
        run["VALUE_NET LOSS"].log(loss)


    run.stop()