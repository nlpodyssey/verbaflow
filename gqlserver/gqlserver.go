package main

//go:generate go run github.com/99designs/gqlgen generate

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/99designs/gqlgen/graphql"
	"github.com/99designs/gqlgen/graphql/handler"
	"github.com/99designs/gqlgen/graphql/handler/extension"
	"github.com/99designs/gqlgen/graphql/handler/lru"
	"github.com/99designs/gqlgen/graphql/handler/transport"
	"github.com/99designs/gqlgen/graphql/playground"
	"github.com/gorilla/websocket"
	"github.com/nlpodyssey/verbaflow/gqlserver/authorization"
	"github.com/nlpodyssey/verbaflow/gqlserver/database"
	"github.com/nlpodyssey/verbaflow/gqlserver/graph"
	corspkg "github.com/rs/cors"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

func main() {
	if err := run(); err != nil {
		log.Fatal().Err(err).Msg("program terminates with error")
	}
}

func run() error {
	flags, err := defineAndParseFlags()
	if err != nil {
		return fmt.Errorf("failed to parse flags: %w", err)
	}

	if err = setupLogger(flags); err != nil {
		return fmt.Errorf("failed to setup logger: %w", err)
	}

	db, err := openAndPrepareDatabase(flags.DBFilename)
	if err != nil {
		return err
	}

	cors := newCORS(strings.Split(" ", flags.CORSOrigins))
	auth := authorization.New(db, flags.CookieHashKey, flags.CookieBlockKey, flags.CookieMaxAge)
	resolver := &graph.Resolver{
		DB: db,
	}

	gqlServer := cors.Handler(
		auth.MiddlewareHandler(
			newGraphqlServer(resolver),
		),
	)

	serverMux := http.NewServeMux()
	serverMux.Handle("/graphql", gqlServer)
	serverMux.Handle("/graphiql", playground.Handler("GraphiQL", "/graphql"))

	httpServer := &http.Server{
		Addr:    flags.ListenAddress,
		Handler: serverMux,
	}

	serveErrChan := make(chan error, 1)
	log.Info().Msg("server listening")
	go func() {
		if flags.TLSEnabled {
			serveErrChan <- httpServer.ListenAndServeTLS(flags.TLSCert, flags.TLSKey)
		} else {
			serveErrChan <- httpServer.ListenAndServe()
		}
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, os.Kill)

	select {
	case err = <-serveErrChan:
		log.Err(err).Msg("failed to serve")
	case sig := <-sigChan:
		log.Warn().Stringer("signal", sig).Msg("received signal")
	}

	signal.Stop(sigChan)
	defer func() {
		close(sigChan)
		close(serveErrChan)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err = httpServer.Shutdown(ctx); err != nil {
		return fmt.Errorf("failed to shutdown server: %w", err)
	}
	return nil
}

func newCORS(allowedOrigins []string) *corspkg.Cors {
	return corspkg.New(corspkg.Options{
		AllowedOrigins:   allowedOrigins,
		AllowedMethods:   []string{http.MethodGet, http.MethodPost},
		AllowedHeaders:   []string{"*"},
		ExposedHeaders:   []string{"*"},
		AllowCredentials: true,
	})
}

type Flags struct {
	ListenAddress  string
	DBFilename     string
	JSONLog        bool
	LogLevel       string
	CORSOrigins    string
	TLSEnabled     bool
	TLSCert        string
	TLSKey         string
	CookieMaxAge   time.Duration
	CookieHashKey  string
	CookieBlockKey string
}

const sampleCookieKey = "01234567890123456789012345678901"

func defineAndParseFlags() (Flags, error) {
	gs := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	var flags Flags
	gs.StringVar(&flags.ListenAddress, "listen", ":8080", "server listening address")
	gs.StringVar(&flags.DBFilename, "db", "verbaflow.sqlite", "SQLite database filename")
	gs.BoolVar(&flags.JSONLog, "json-log", false, "log messages in JSON format")
	gs.StringVar(&flags.LogLevel, "log-level", "info", "log level")
	gs.StringVar(&flags.CORSOrigins, "cors-origins", "*", "space-separated list of allowed origins")
	gs.BoolVar(&flags.TLSEnabled, "tls", false, "enable TLS")
	gs.StringVar(&flags.TLSCert, "tls-cert", "", "TLS certificate (ignored if TLS is disabled)")
	gs.StringVar(&flags.TLSKey, "tls-key", "", "TLS private key (ignored if TLS is disabled)")
	gs.DurationVar(&flags.CookieMaxAge, "cookie-max-age", 2*time.Hour, "secure cookie max age")
	gs.StringVar(&flags.CookieHashKey, "cookie-hash-key", sampleCookieKey, "secure cookie hash key")
	gs.StringVar(&flags.CookieBlockKey, "cookie-block-key", sampleCookieKey, "secure cookie block key")
	err := gs.Parse(os.Args[1:])
	return flags, err
}

func setupLogger(flags Flags) error {
	if !flags.JSONLog {
		log.Logger = log.Logger.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	}

	level, err := zerolog.ParseLevel(flags.LogLevel)
	if err != nil {
		return fmt.Errorf("invalid log level %q: %w", flags.LogLevel, err)
	}
	log.Logger = log.Logger.Level(level)

	return nil
}

func openAndPrepareDatabase(filename string) (*gorm.DB, error) {
	db, err := gorm.Open(sqlite.Open(filename), &gorm.Config{
		Logger: database.NewLogger(log.Logger),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	if err = db.AutoMigrate(database.Models...); err != nil {
		return nil, fmt.Errorf("failed to auto-migrate database: %w", err)
	}

	if err = createAdminUserIfNoUsers(db); err != nil {
		return nil, err
	}

	return db, nil
}

func createAdminUserIfNoUsers(db *gorm.DB) error {
	var usersCount int64
	if err := db.Model(&database.User{}).Count(&usersCount).Error; err != nil {
		return fmt.Errorf("failed to count database users: %w", err)
	}
	if usersCount != 0 {
		return nil
	}

	passwordHash, err := bcrypt.GenerateFromPassword([]byte("admin"), bcrypt.DefaultCost)
	if err != nil {
		return fmt.Errorf("failed to generate admin password hash: %w", err)
	}
	admin := database.User{
		Username:     "admin",
		PasswordHash: string(passwordHash),
		IsAdmin:      true,
	}
	if err = db.Create(&admin).Error; err != nil {
		return fmt.Errorf("failed to create admin user: %w", err)
	}
	return nil
}

func newGraphqlServer(resolver *graph.Resolver) *handler.Server {
	server := handler.New(graph.NewExecutableSchema(graph.Config{Resolvers: resolver}))

	server.AroundOperations(func(ctx context.Context, next graphql.OperationHandler) graphql.ResponseHandler {
		oc := graphql.GetOperationContext(ctx)
		log.Trace().Str("operation-name", oc.OperationName).Str("query", oc.RawQuery).Any("variables", oc.Variables).Msg("GraphQL query")
		return next(ctx)
	})

	server.AddTransport(transport.Options{})
	server.AddTransport(transport.GET{})
	server.AddTransport(transport.POST{})
	server.AddTransport(transport.MultipartForm{})

	server.SetQueryCache(lru.New(1000))

	server.Use(extension.Introspection{})
	server.Use(extension.AutomaticPersistedQuery{
		Cache: lru.New(100),
	})

	server.AddTransport(&transport.Websocket{
		KeepAlivePingInterval: 10 * time.Second,
		Upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				// Origin already validated by server-wide CORS handler
				return true
			},
		},
	})
	return server
}
