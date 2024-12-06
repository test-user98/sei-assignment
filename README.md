run this to start server: `uvicorn app:app --reload`

USE below curl to train and test:

**Below curl is used to add the policy doc.**


```
curl -X POST "http://127.0.0.1:8000/train" \
-H "Authorization: Bearer e680e668-363f-4c5f-90ff-44fcd1e3a84e" \
-H "Content-Type: application/json" \
-d '{
  "url": "https://docs.stripe.com/treasury/marketing-treasury#terms-to-avoid",
  "policy_name": "stripe_policy"
}'
```              

**TO Test a document against a benchmarked trained document:**

```
curl -X POST "http://127.0.0.1:8000/test" \
-H "Authorization: Bearer e680e668-363f-4c5f-90ff-44fcd1e3a84e" \
-H "Content-Type: application/json" \
-d '{
  "url": "https://mercury.com/"
}'
```
