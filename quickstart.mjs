// Zernio quickstart: profile -> connect URL -> accounts -> schedule post -> status.
// Reads the API key from ZERNIO_API_KEY. Never hardcode it.
import Zernio from '@zernio/node';
import { createInterface } from 'node:readline/promises';

if (!process.env.ZERNIO_API_KEY) {
  console.error('Set ZERNIO_API_KEY first: export ZERNIO_API_KEY="sk_..."');
  process.exit(1);
}

const PLATFORM = 'instagram';
const zernio = new Zernio(); // reads ZERNIO_API_KEY

// Stop on any API error so nothing runs on bad state.
function check(step, res) {
  if (res.error) {
    console.error(`${step} failed:`, JSON.stringify(res.error, null, 2));
    process.exit(1);
  }
  return res.data;
}

// 1. A profile groups accounts.
const created = check('createProfile', await zernio.profiles.createProfile({
  body: { name: 'Majed Photos' }
}));
const profileId = created.profile._id;

// 2. Get the URL where the account owner authorizes Zernio.
const connect = check('getConnectUrl', await zernio.connect.getConnectUrl({
  path: { platform: PLATFORM },
  query: { profileId }
}));

console.log('Open this URL and approve:', connect.authUrl);
const prompt = createInterface({ input: process.stdin, output: process.stdout });
await prompt.question('Press Enter once the account is connected ');
prompt.close();

// 3. Posts target accounts by _id, not by username.
const listed = check('listAccounts', await zernio.accounts.listAccounts());
const account = listed.accounts.find((a) => a.platform === PLATFORM);
if (!account) {
  console.error(`No ${PLATFORM} account connected yet.`);
  process.exit(1);
}

// 4. Schedule the post. scheduledFor is read in the timezone you send.
const scheduled = check('createPost', await zernio.posts.createPost({
  body: {
    content: 'Hello world. This is my first post from the Zernio API.',
    scheduledFor: '2027-01-01T12:00:00',
    timezone: 'Asia/Riyadh',
    platforms: [{ platform: PLATFORM, accountId: account._id }]
  }
}));
const postId = scheduled.post._id;

// 5. Read it back: scheduled now, published after the scheduled time.
const fetched = check('getPost', await zernio.posts.getPost({ path: { postId } }));
console.log(fetched.post.status);
