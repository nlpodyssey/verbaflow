package database

import (
	"context"
	"time"

	"github.com/rs/zerolog"
	gormlogger "gorm.io/gorm/logger"
)

type Logger struct {
	zerolog.Logger
}

var _ gormlogger.Interface = Logger{}

func NewLogger(parent zerolog.Logger) Logger {
	return Logger{Logger: parent}
}

var gormToZeroLogLevel = map[gormlogger.LogLevel]zerolog.Level{
	gormlogger.Silent: zerolog.Disabled,
	gormlogger.Error:  zerolog.ErrorLevel,
	gormlogger.Warn:   zerolog.WarnLevel,
	gormlogger.Info:   zerolog.InfoLevel,
}

func (l Logger) LogMode(level gormlogger.LogLevel) gormlogger.Interface {
	var zeroLevel zerolog.Level
	switch {
	case level < gormlogger.Silent:
		zeroLevel = zerolog.Disabled
	case level > gormlogger.Info:
		zeroLevel = zerolog.TraceLevel
	default:
		zeroLevel = gormToZeroLogLevel[level]
	}
	return Logger{Logger: l.Logger.Level(zeroLevel)}
}

func (l Logger) Info(_ context.Context, msg string, data ...interface{}) {
	l.Logger.Info().Msgf(msg, data...)
}

func (l Logger) Warn(_ context.Context, msg string, data ...interface{}) {
	l.Logger.Warn().Msgf(msg, data...)
}

func (l Logger) Error(_ context.Context, msg string, data ...interface{}) {
	l.Logger.Error().Msgf(msg, data...)
}

func (l Logger) Trace(_ context.Context, begin time.Time, fc func() (sql string, rowsAffected int64), err error) {
	switch {
	case err != nil && l.GetLevel() <= zerolog.ErrorLevel:
		elapsed := time.Since(begin)
		sql, rows := fc()
		l.Err(err).Str("sql", sql).Int64("rows", rows).Dur("elapsed", elapsed).Msg("query error")
	case l.GetLevel() <= zerolog.DebugLevel:
		elapsed := time.Since(begin)
		sql, rows := fc()
		l.Debug().Str("sql", sql).Int64("rows", rows).Dur("elapsed", elapsed).Msg("query")
	}
}
