from .random import Random
from .recommender import Recommender
from dataclasses import dataclass
import numpy as np


def _get_users_pics(user: int, prev_track: int, redis, catalog):
    user_history = redis.get(user)
    if user_history is None:
        user_history = [prev_track]
        redis.set(user, catalog.to_bytes(user_history))
    else:
        user_history = catalog.from_bytes(user_history)
        user_history = list(user_history)
    return user_history


def _choose_and_update_pics(recommendations, user, users_pics, redis, catalog):
    x = int(np.random.choice(recommendations, 1)[0])
    users_pics.append(x)
    redis.set(user, catalog.to_bytes(users_pics))
    return x


class ContextualTracks(Recommender):
    def __init__(self, tracks_redis, users_history_redis, catalog):
        self.tracks_redis = tracks_redis
        self.users_history_redis = users_history_redis
        self.catalog = catalog
        self.fallback = Random(tracks_redis)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        previous_track = self.tracks_redis.get(prev_track)
        if previous_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        previous_track = self.catalog.from_bytes(previous_track)
        recommendations = previous_track.recommendations
        if not recommendations:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        recommendations = list(recommendations)

        user_history = _get_users_pics(user, prev_track, self.users_history_redis, self.catalog)
        user_history = list(user_history)

        recommendations = list(set(recommendations) - set(user_history))
        if len(recommendations) == 0:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        return _choose_and_update_pics(recommendations, user, user_history, self.users_history_redis, self.catalog)


class ContextualUsers(Recommender):
    def __init__(self, tracks_redis, users_redis, users_history_redis, catalog):
        self.users_redis = users_redis
        self.users_history_redis = users_history_redis
        self.catalog = catalog
        self.fallback = Random(tracks_redis)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recommendations = self.users_redis.get(user)
        if recommendations is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        recommendations = self.catalog.from_bytes(recommendations)
        recommendations = list(recommendations)

        user_history = _get_users_pics(user, prev_track, self.users_history_redis, self.catalog)

        recommendations = list(set(recommendations) - set(user_history))
        if len(recommendations) == 0:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        recommendations = list(recommendations)

        return _choose_and_update_pics(recommendations, user, user_history, self.users_history_redis, self.catalog)


class ContextualBest(Recommender):
    def __init__(self, tracks_redis, users_redis, users_history_redis, catalog):
        self.tracks_redis = tracks_redis
        self.users_redis = users_redis
        self.users_history_redis = users_history_redis
        self.catalog = catalog
        self.fallback = Random(tracks_redis)
        self.users_recommender = ContextualUsers(tracks_redis, users_redis, users_history_redis, catalog)
        self.tracks_recommender = ContextualTracks(tracks_redis, users_history_redis, catalog)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        previous_track = self.tracks_redis.get(prev_track)
        if previous_track is None:
            return self.users_recommender.recommend_next(user, prev_track, prev_track_time)

        previous_track = self.catalog.from_bytes(previous_track)
        recommendations_tracks = previous_track.recommendations
        if not recommendations_tracks:
            return self.users_recommender.recommend_next(user, prev_track, prev_track_time)
        recommendations_tracks = list(recommendations_tracks)

        recommendations_users = self.users_redis.get(user)
        if recommendations_users is None:
            return self.tracks_recommender.recommend_next(user, prev_track, prev_track_time)
        recommendations_users = self.catalog.from_bytes(recommendations_users)
        recommendations_users = list(recommendations_users)

        user_history = _get_users_pics(user, prev_track, self.users_history_redis, self.catalog)

        recommendations = list(set(recommendations_tracks) & set(recommendations_users) - set(user_history))
        if len(recommendations) == 0:
            if np.random.rand() < 0.5:
                return self.tracks_recommender.recommend_next(user, prev_track, prev_track_time)
            else:
                return self.users_recommender.recommend_next(user, prev_track, prev_track_time)

        return _choose_and_update_pics(recommendations, user, user_history, self.users_history_redis, self.catalog)


@dataclass
class _Info:
    track: int
    time: float
    recommender: str
    changed: bool


class ContextualSmart(ContextualBest):
    def __init__(self, tracks_redis, users_redis, users_history_redis, catalog):
        super().__init__(tracks_redis, users_redis, users_history_redis, catalog)
        self.last_track_info_by_user = {}

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        previous_track = self.tracks_redis.get(prev_track)
        if previous_track is None:
            return self.users_recommender.recommend_next(user, prev_track, prev_track_time)

        previous_track = self.catalog.from_bytes(previous_track)
        recommendations_tracks = previous_track.recommendations
        if not recommendations_tracks:
            return self.users_recommender.recommend_next(user, prev_track, prev_track_time)
        recommendations_tracks = list(recommendations_tracks)

        recommendations_users = self.users_redis.get(user)
        if recommendations_users is None:
            return self.tracks_recommender.recommend_next(user, prev_track, prev_track_time)
        recommendations_users = self.catalog.from_bytes(recommendations_users)
        recommendations_users = list(recommendations_users)

        user_history = _get_users_pics(user, prev_track, self.users_history_redis, self.catalog)

        recommendations = list(set(recommendations_tracks) & set(recommendations_users) - set(user_history))
        if len(recommendations) == 0:
            if user not in self.last_track_info_by_user:
                self.last_track_info_by_user[user] = _Info(prev_track, 1.0, ContextualSmart.__name__, False)

            info = self.last_track_info_by_user[user]

            if prev_track_time < 0.75 <= info.time and not info.changed:
                info.changed = True
                if info.recommender == ContextualTracks.__name__:
                    return self.users_recommender.recommend_next(user, prev_track, prev_track_time)
                elif info.recommender == ContextualUsers.__name__:
                    return self.tracks_recommender.recommend_next(user, prev_track, prev_track_time)

            info.track = prev_track
            info.time = prev_track_time
            if np.random.rand() < 0.5:
                info.recommender = ContextualTracks.__name__
                return self.tracks_recommender.recommend_next(user, prev_track, prev_track_time)
            else:
                info.recommender = ContextualUsers.__name__
                return self.users_recommender.recommend_next(user, prev_track, prev_track_time)

        self.last_track_info_by_user[user] = _Info(prev_track, prev_track_time, ContextualSmart.__name__, False)
        return _choose_and_update_pics(recommendations, user, user_history, self.users_history_redis, self.catalog)
