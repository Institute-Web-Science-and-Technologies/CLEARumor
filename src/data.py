"""Functions for loading the RumorEval dataset.

Requires that the source files are manually placed in the `EXTERNAL_DATA_DIR`
folder. See the README for details.

Data is read directly from the .zip-files without extracting them, because this
was deemed more elegant.
"""

import json
from enum import Enum
from itertools import chain
from pathlib import Path
from sys import exit
from time import time
from typing import Dict, List, Optional
from zipfile import ZipFile

DATA_DIR = Path('data')

EXTERNAL_DATA_DIR = DATA_DIR / 'external'
ELMO_WEIGHTS_FILE = EXTERNAL_DATA_DIR / 'elmo_2x4096_512_2048cnn_2xhighway' \
                                        '_5.5B_weights.hdf5'
ELMO_OPTIONS_FILE = EXTERNAL_DATA_DIR / 'elmo_2x4096_512_2048cnn_2xhighway' \
                                        '_5.5B_options.json'
TRAINING_DATA_ARCHIVE_FILE = EXTERNAL_DATA_DIR / 'rumoureval-2019' \
                                                 '-training-data.zip'
TEST_DATA_ARCHIVE_FILE = EXTERNAL_DATA_DIR / 'rumoureval-2019' \
                                             '-test-data.zip'
EVALUATION_DATA_FILE = EXTERNAL_DATA_DIR / 'final-eval-key.json'
EVALUATION_SCRIPT_FILE = EXTERNAL_DATA_DIR / 'home_scorer_macro.py'


class Post:
    """Data class for both Twitter and Reddit posts.

    Args:
        id: ID of the post.
        text: Text of the post for Twitter, title/body of the post for Reddit.
        depth: Depth in the thread. Source posts always have `depth=0`, replies
            to source posts have `depth=1`, replies to replies have `depth=2`,
            and so forth.
        source: Whether the post is from Twitter or from Reddit.
        has_media: `True` if the posts links to any media, `False` otherwise.
        source_id: The ID of the source post of the thread. If the current post
            is itself a source post, this is equal to `self.id`.
        topic: The rumor topic the posts belongs to for Twitter. `None` for
            Reddit posts, since the dataset has no topic labels for them.
        user_verified: Whether the user is a verified Twitter user. `None` for
            Reddit posts, since the dataset does not contain any info on this.
        followers_count: The number of followers for Twitter posts' authors.
            `None` for Reddit posts, since the concept doesn't exist for Reddit.
        friends_count: The number of friends for Twitter posts' authors.
            `None` for Reddit posts, since the concept doesn't exist for Reddit.
        upvote_ratio: The upvote ratio for Reddit posts. `None` for Twitter
            posts, since the concept doesn't exist for Twitter.
    """

    class PostSource(Enum):
        """Enum to designate whether a posts is from Twitter or from Reddit."""
        twitter = 1
        reddit = 2

    def __init__(self,
                 id: str,
                 text: str,
                 depth: int,
                 source: PostSource,
                 has_media: bool,
                 source_id: Optional[str] = None,
                 topic: Optional[str] = None,
                 user_verified: Optional[bool] = None,
                 followers_count: Optional[int] = None,
                 friends_count: Optional[int] = None,
                 upvote_ratio: Optional[float] = None):
        self.id = id
        self.text = text
        self.depth = depth
        self.source = source
        self.has_media = has_media
        self.source_id = source_id or self.id
        self.topic = topic
        self.user_verified = user_verified
        self.followers_count = followers_count
        self.friends_count = friends_count
        self.upvote_ratio = upvote_ratio

    @property
    def has_source_depth(self) -> bool:
        """Whether the post is the source of a thread."""
        return self.depth == 0

    @property
    def has_reply_depth(self) -> bool:
        """Whether the post is a reply to the source of a thread."""
        return self.depth == 1

    @property
    def has_rreply_depth(self) -> bool:
        """Whether the post is neither source nor reply to a thread's source."""
        return self.depth >= 2

    @property
    def url(self) -> str:
        """Url of the post (useful for debugging)."""
        if self.source == self.PostSource.twitter:
            return 'https://twitter.com/statuses/{}'.format(self.id)
        elif self.source == self.PostSource.reddit:
            if self.source_id == self.id:
                return 'https://reddit.com//comments/{}'.format(self.id)
            return 'https://reddit.com//comments/{}//{}'.format(self.source_id,
                                                                self.id)
        raise ValueError('Invalid post source value, must be either Twitter or '
                         'Reddit.')

    def __repr__(self) -> str:
        return 'Post {}'.format(vars(self))

    @classmethod
    def load_from_twitter_dict(cls,
                               twitter_dict: Dict,
                               post_depths: Dict[str, int],
                               source_id: Optional[str] = None,
                               topic: Optional[str] = None) -> 'Post':
        """Creates a `Post` instance from a JSON dict of a Twitter post.

        Args:
            twitter_dict: The JSON dict.
            post_depths: A map that gives the depth of the post by it's ID.
            source_id: The ID of the thread's source post. `None` if this post
                is itself the source post.
            topic: The rumor topic the posts is labelled to belong to.

        Returns:
            The created `Post` instance.
        """
        id = twitter_dict['id_str']
        return Post(id=id,
                    text=twitter_dict['text'],
                    depth=post_depths[id],
                    source=cls.PostSource.twitter,
                    has_media='media' in twitter_dict['entities'],
                    source_id=source_id,
                    topic=topic,
                    user_verified=twitter_dict['user']['verified'],
                    followers_count=twitter_dict['user']['followers_count'],
                    friends_count=twitter_dict['user']['friends_count'])

    @classmethod
    def load_from_reddit_dict(cls,
                              reddit_dict: Dict,
                              post_depths: Dict[str, int],
                              source_id: Optional[str] = None) -> 'Post':
        """Creates a `Post` instance from a JSON dict of a Reddit post.

        For some reason, the JSON files for 19 Reddit reply posts from the
        training dataset (with source IDs either "49l01s" or "8j9s33") and 71
        reply posts from the testing dataset (with source ID "7imq99") contain
        only their ID data, and are missing all other data. Since these post are
        still referenced in the labeled datasets, we just return them with an
        empty text (`''`).

        Args:
            reddit_dict: The JSON dict.
            post_depths: A map that gives the depth of the post by it's ID.
            source_id: The ID of the thread's source post. `None` if this post
                is itself the source post.

        Returns:
            The created `Post` instance.
        """
        data = reddit_dict['data']
        if 'children' in data and isinstance(data['children'][0], dict):
            data = data['children'][0]['data']

        id = data['id']
        return Post(id=id,
                    text=data.get('title') or data.get('body') or '',
                    depth=post_depths[id],
                    source=cls.PostSource.reddit,
                    has_media=('domain' in data
                               and not data['domain'].startswith('self.')),
                    source_id=source_id,
                    upvote_ratio=data.get('upvote_ratio'))


def check_for_required_external_data_files() -> None:
    """Checks whether all required external data files are present.

    If not, will print a message to stderr and exit.
    """
    for required_file in [ELMO_WEIGHTS_FILE, ELMO_OPTIONS_FILE,
                          TRAINING_DATA_ARCHIVE_FILE, TEST_DATA_ARCHIVE_FILE,
                          EVALUATION_SCRIPT_FILE]:
        if not required_file.exists():
            exit('Required file "{}" is not present. See the README on how to '
                 'obtain it.'.format(required_file))


def load_posts() -> Dict[str, Post]:
    """Loads all Twitter and Reddit posts into a dictionary.

    Since the dataset is very small, we just load the whole dataset into RAM.

    Returns:
        A dictionary mapping post IDs to their respective posts.
    """

    def get_archive_directory_structure(archive: ZipFile) -> Dict:
        """Parses a ZipFile's list of files into a hierarchical representation.

        We need to do this because ZipFile just gives us a list of all files in
        contains and doesn't provide any methods to check which files lie in a
        specific subdirectory.

        Args:
            archive: The archive to parse.

        Returns:
            A nested dictionary. Keys of this dictionary are either file names
            which point to their full path in the archive or directory names
            which again point to a nested dictionary that contains their
            contents.

        Example:
            If the archive would contain the following files::

                ['foo.txt',
                 'bar/bar.log',
                 'bar/baz.out',
                 'bar/boogy/text.out']

            This would be transformed into the following hierarchical form::

                {
                    'foo.txt': 'foo.txt',
                    'bar': {
                        'bar.log': 'bar/bar.log',
                        'baz.out': 'bar/baz.out',
                        'boogy': {
                            'text.out': 'bar/boogy/text.out'
                        }
                    }
                }
        """
        result = {}
        for file in archive.namelist():
            # Skip directories in archive.
            if file.endswith('/'):
                continue

            d = result
            path = file.split('/')[1:]  # [1:] to skip top-level directory.
            for p in path[:-1]:  # [:-1] to skip filename
                if p not in d:
                    d[p] = {}
                d = d[p]
            d[path[-1]] = file
        return result

    def calc_post_depths_from_thread_structure(thread_structure: Dict) \
            -> Dict[str, int]:
        """Calculates the nested depth of each post in a thread.

        We determine post depth from the provided `structure.json` files in the
        dataset because this is easier than following the chain of a post's
        parents to the source post of a thread.

        Args:
            thread_structure: The parsed JSON dict from one of the dataset's
                `structure.json` files.

        Returns:
            A dictionary mapping post IDs to their nested depth. The source
            post of a thread always has depth `0`, first level replies `1`, etc.

        Example:
            If the `thread_structure` would look like the following::

                {
                    'foo': {
                        'bar': [],
                        'baz': {
                            'boogy': []
                        },
                        'qux': []
                    }
                }

            The parsed post depths would be::

                {
                    'foo': 0,
                    'bar': 1,
                    'baz': 1,
                    'boogy': 2,
                    'qux': 1
                }
        """
        post_depths = {}

        def walk(thread: Dict, depth: int) -> None:
            for post_id, subthread in thread.items():
                post_depths[post_id] = depth
                if isinstance(subthread, Dict):
                    walk(subthread, depth + 1)

        walk(thread_structure, 0)
        return post_depths

    training_data_archive = ZipFile(TRAINING_DATA_ARCHIVE_FILE)
    training_data_contents = get_archive_directory_structure(
        training_data_archive)
    twitter_english = training_data_contents['twitter-english']
    reddit_training_data = training_data_contents['reddit-training-data']
    reddit_dev_data = training_data_contents['reddit-dev-data']

    test_data_archive = ZipFile(TEST_DATA_ARCHIVE_FILE)
    test_data_contents = get_archive_directory_structure(test_data_archive)
    twitter_en_test_data = test_data_contents['twitter-en-test-data']
    reddit_test_data = test_data_contents['reddit-test-data']

    posts: Dict[str, Post] = {}

    # -- Load Twitter posts ----------------------------------------------------
    for archive, topics in [(training_data_archive, twitter_english.items()),
                            (test_data_archive, twitter_en_test_data.items())]:
        for topic, threads in topics:
            for thread in threads.values():
                post_depths = calc_post_depths_from_thread_structure(
                    json.loads(archive.read(thread['structure.json'])))

                source_post = Post.load_from_twitter_dict(
                    json.loads(archive.read(
                        next(iter(thread['source-tweet'].values())))),
                    post_depths,
                    topic=topic)
                posts[source_post.id] = source_post

                for reply in thread.get('replies', {}).values():
                    reply_post = Post.load_from_twitter_dict(
                        json.loads(archive.read(reply)),
                        post_depths,
                        source_id=source_post.id,
                        topic=topic)
                    posts[reply_post.id] = reply_post

    # -- Load Reddit posts. ----------------------------------------------------
    for archive, threads in [(training_data_archive,
                              chain(reddit_training_data.values(),
                                    reddit_dev_data.values())),
                             (test_data_archive, reddit_test_data.values())]:
        for thread in threads:
            post_depths = calc_post_depths_from_thread_structure(
                json.loads(archive.read(thread['structure.json'])))

            source_post = Post.load_from_reddit_dict(
                json.loads(archive.read(
                    next(iter(thread['source-tweet'].values())))),
                post_depths)
            posts[source_post.id] = source_post

            for reply in thread.get('replies', {}).values():
                reply_post = Post.load_from_reddit_dict(
                    json.loads(archive.read(reply)),
                    post_depths,
                    source_id=source_post.id)
                posts[reply_post.id] = reply_post

    return posts


class SdqcInstance:
    """Data class for SDQC (RumorEval Task A) instances.

    Args:
        post_id: An ID referencing a Twitter or a Reddit post.
        label: A label whether the stance expressed in the referenced post is
            *support*, *deny*, *query*, or *comment* towards the rumor expressed
            in the thread's source post.
    """

    class SdqcLabel(Enum):
        """Enum for SDQC labels `support`, `deny`, `query`, and `comment`."""
        support = 1
        deny = 2
        query = 3
        comment = 4

    def __init__(self, post_id: str, label: SdqcLabel):
        self.post_id = post_id
        self.label = label

    def __repr__(self):
        return 'SDQC ({}, {})'.format(self.post_id, self.label)


def load_sdcq_instances() -> (List[SdqcInstance],
                              List[SdqcInstance],
                              Optional[List[SdqcInstance]]):
    """Load SDQC (RumorEval Task A) training, dev, and test datasets.

    Returns:
        A tuple containing lists of SDQC instances. The first element is the
        training dataset, the second the dev, and the third the test, if it is
        available, otherwise `None`.
    """

    def load_from_json_dict(json_dict: Dict[str, Dict[str, str]]) \
            -> List[SdqcInstance]:
        return [SdqcInstance(post_id, SdqcInstance.SdqcLabel[label])
                for post_id, label in json_dict['subtaskaenglish'].items()]

    training_data_archive = ZipFile(TRAINING_DATA_ARCHIVE_FILE)
    sdqc_train = load_from_json_dict(json.loads(training_data_archive.read(
        'rumoureval-2019-training-data/train-key.json')))
    sdqc_dev = load_from_json_dict(json.loads(training_data_archive.read(
        'rumoureval-2019-training-data/dev-key.json')))
    sdqc_test = None

    if EVALUATION_DATA_FILE.exists():
        with EVALUATION_DATA_FILE.open('rb') as fin:
            sdqc_test = load_from_json_dict(json.loads(fin.read()))

    return sdqc_train, sdqc_dev, sdqc_test


class VerifInstance:
    """Data class for Verification (RumorEval Task B) instances.

    Args:
        post_id: An ID referencing a Twitter or a Reddit thread's source post.
        label: A label whether the rumor expressed in the referenced post is
            `true`, `false`, or `unverified`.
    """

    class VerifLabel(Enum):
        """ Enum for verification labels `true`, `false`, and `unverified`."""
        true = 1
        false = 2
        unverified = 3

    def __init__(self, post_id: str, label: VerifLabel):
        self.post_id = post_id
        self.label = label

    def __str__(self):
        print('Verif ({}, {})'.format(self.post_id, self.label))


def load_verif_instances() -> (List[VerifInstance],
                               List[VerifInstance],
                               Optional[List[VerifInstance]]):
    """Load Verification (RumorEval Task B) training, dev, and test datasets.

    Returns:
        A tuple containing lists of Verification instances. The first element is
        the training dataset, the second the dev, and the third the test, if it
        is available, otherwise `None`.
    """

    def load_from_json_dict(json_dict: Dict[str, Dict[str, str]]) \
            -> List[VerifInstance]:
        return [VerifInstance(post_id, VerifInstance.VerifLabel[label])
                for post_id, label in json_dict['subtaskbenglish'].items()]

    training_data_archive = ZipFile(TRAINING_DATA_ARCHIVE_FILE)
    verif_train = load_from_json_dict(json.loads(training_data_archive.read(
        'rumoureval-2019-training-data/train-key.json')))
    verif_dev = load_from_json_dict(json.loads(training_data_archive.read(
        'rumoureval-2019-training-data/dev-key.json')))
    verif_test = None

    if EVALUATION_DATA_FILE.exists():
        with EVALUATION_DATA_FILE.open('rb') as fin:
            verif_test = load_from_json_dict(json.loads(fin.read()))

    return verif_train, verif_dev, verif_test


def load() -> (Dict[str, Post],
               List[SdqcInstance],
               List[SdqcInstance],
               Optional[List[SdqcInstance]],
               List[VerifInstance],
               List[VerifInstance],
               Optional[List[VerifInstance]]):
    """Load all Twitter/Reddit posts and SDQC and Verification datasets.

    See the `load_posts()`, `load_sdqc_instances()`, and
    `load_verif_instances()` for further documentation.

    Also prints a short analysis of the dataset.

    Returns:
        A tuple containing: (1) a dictionary mapping post IDs to post objects,
        the SDQC (2) train, (3) dev, and (4) test datasets, the Verification
        (5) train, (6) dev, and (7) test datasets.
    """
    check_for_required_external_data_files()

    print('Loading posts...')
    time_before = time()
    posts = load_posts()
    time_after = time()
    print('  Took {:.2f}s.'.format(time_after - time_before))
    print('  Number of posts: {:d} (Reddit={:d}, Twitter={:d})'.format(
        len(posts),
        sum(1 for p in posts.values() if p.source == Post.PostSource.reddit),
        sum(1 for p in posts.values() if p.source == Post.PostSource.twitter)))

    print('Loading instances...')
    time_before = time()
    sdqc_train, sdqc_dev, sdqc_test = load_sdcq_instances()
    verif_train, verif_dev, verif_test = load_verif_instances()
    time_after = time()
    print('  Took {:.2f}s'.format(time_after - time_before))
    print('  Number of SDQC instances: train={:d}, dev={:d}, test={:d}'.format(
        len(sdqc_train), len(sdqc_dev), len(sdqc_test) if sdqc_test else 0))
    print('  Number of Verif instances: train={:d}, dev={:d}, test={:d}'.format(
        len(verif_train), len(verif_dev), len(verif_test) if verif_test else 0))

    return (posts,
            sdqc_train, sdqc_dev, sdqc_test,
            verif_train, verif_dev, verif_test)
