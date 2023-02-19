package authorization

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/securecookie"
	"github.com/nlpodyssey/verbaflow/gqlserver/database"
	"github.com/rs/zerolog/log"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"
)

type ctxKey string

const (
	authKey               = ctxKey("auth")
	userCtxKey            = ctxKey("user")
	responseWriterKey     = ctxKey("responseWriter")
	unauthorizedResponse  = `{"errors":[{"message":"Unauthorized"}]}`
	internalErrorResponse = `{"errors":[{"message":"Internal server error"}]}`
	cookieName            = "verbaflow"
)

func UserForContext(ctx context.Context) *database.User {
	user, ok := ctx.Value(userCtxKey).(*database.User)
	if !ok {
		log.Warn().Msg("user not found in context")
		return nil
	}
	return user
}

type Auth struct {
	db           *gorm.DB
	secureCookie *securecookie.SecureCookie
	maxAge       int
}

func New(db *gorm.DB, hashKey, blockKey string, maxAge time.Duration) *Auth {
	return &Auth{
		db:           db,
		secureCookie: securecookie.New([]byte(hashKey), []byte(blockKey)),
		maxAge:       int(maxAge.Seconds()),
	}
}

func (auth *Auth) MiddlewareHandler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()

		user, err := auth.resolveUserFromBasicAuth(r)
		if err != nil {
			log.Err(err).Msg("basic auth failed")
			http.Error(w, internalErrorResponse, http.StatusInternalServerError)
			return
		}
		if user == nil {
			user, err = auth.resolveUserFromCookie(r)
			if err != nil {
				log.Err(err).Msg("cookie authentication failed")
				http.Error(w, internalErrorResponse, http.StatusInternalServerError)
				return
			}
		}

		if user == nil {
			http.Error(w, unauthorizedResponse, http.StatusUnauthorized)
			return
		}

		ctx = context.WithValue(ctx, responseWriterKey, w)
		ctx = context.WithValue(ctx, authKey, auth)
		ctx = context.WithValue(ctx, userCtxKey, user)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func SignIn(ctx context.Context) (*database.User, error) {
	user, ok := ctx.Value(userCtxKey).(*database.User)
	if !ok || user == nil {
		return nil, nil
	}
	w := ctx.Value(responseWriterKey).(http.ResponseWriter)
	auth := ctx.Value(authKey).(*Auth)

	value := map[string]string{"UserID": strconv.FormatUint(uint64(user.ID), 10)}
	encoded, err := auth.secureCookie.Encode(cookieName, value)
	if err != nil {
		return nil, fmt.Errorf("failed to encode cookie: %w", err)
	}
	cookie := newCookie(encoded, auth.maxAge)
	http.SetCookie(w, cookie)
	return user, nil
}

func SignOut(ctx context.Context) {
	w := ctx.Value(responseWriterKey).(http.ResponseWriter)
	cookie := newCookie("", -1)
	http.SetCookie(w, cookie)
}

func newCookie(value string, maxAge int) *http.Cookie {
	return &http.Cookie{
		Name:     cookieName,
		Value:    value,
		Path:     "/",
		Secure:   true,
		HttpOnly: true,
		SameSite: http.SameSiteNoneMode,
		MaxAge:   maxAge,
	}
}

func (auth *Auth) resolveUserFromBasicAuth(r *http.Request) (*database.User, error) {
	username, pass, ok := r.BasicAuth()
	if !ok || strings.TrimSpace(username) == "" || strings.TrimSpace(pass) == "" {
		return nil, nil
	}

	var user *database.User
	res := auth.db.Limit(1).Find(&user, "username = ?", username)
	if err := res.Error; err != nil {
		return nil, fmt.Errorf("failed to query user by username: %w", err)
	}
	if res.RowsAffected == 0 {
		return nil, nil
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(pass)); err != nil {
		return nil, nil
	}
	return user, nil
}

func (auth *Auth) resolveUserFromCookie(r *http.Request) (*database.User, error) {
	cookie, err := r.Cookie(cookieName)
	if err != nil {
		if errors.Is(err, http.ErrNoCookie) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get cookie: %w", err)
	}

	value := make(map[string]string)
	err = auth.secureCookie.Decode(cookieName, cookie.Value, &value)
	if err != nil {
		log.Warn().Err(err).Msg("failed to decode cookie")
		return nil, nil
	}

	var user *database.User
	res := auth.db.Limit(1).Find(&user, "id = ?", value["UserID"])
	if err = res.Error; err != nil {
		return nil, fmt.Errorf("failed to query user by ID: %w", err)
	}
	if res.RowsAffected == 0 {
		return nil, nil
	}
	return user, nil
}
